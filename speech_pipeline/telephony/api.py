"""Flask blueprint for the telephony API.

Endpoints
---------

Admin-only (``--admin-token``):

- ``PUT    /api/pbx/<id>``             – register / update PBX
- ``GET    /api/pbx``                  – list PBX connections
- ``DELETE /api/pbx/<id>``             – remove PBX
- ``PUT    /api/accounts/<id>``        – register / update account
- ``GET    /api/accounts``             – list accounts
- ``GET    /api/accounts/<id>``        – get account
- ``DELETE /api/accounts/<id>``        – delete account + its subscribers

Account-scoped (account token **or** admin):

- ``PUT    /api/subscribe/<id>``       – register / refresh subscriber
- ``GET    /api/subscribers``          – list own subscribers
- ``GET    /api/subscribers/<id>``     – get own subscriber
- ``DELETE /api/subscribers/<id>``     – unsubscribe
- ``POST   /api/calls``               – create a call (conference)
- ``GET    /api/calls``               – list own calls
- ``GET    /api/calls/<call_id>``     – get call details
- ``POST   /api/calls/<call_id>/commands`` – send commands to a call
- ``DELETE /api/calls/<call_id>``     – end a call
- ``POST   /api/nonce``               – create webclient nonce
- ``GET    /api/nonces``              – list own nonces
- ``DELETE /api/nonce/<nonce>``       – revoke nonce
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from flask import Blueprint, g, jsonify, request

from . import auth, call_state, pbx, subscriber

_LOGGER = logging.getLogger("telephony.api")

api = Blueprint("telephony_api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _account_id() -> Optional[str]:
    """Return current account id, or None for admin."""
    acct = getattr(g, "account", None)
    return acct["id"] if acct else None


def _account_pbx() -> Optional[str]:
    """Return PBX this account is pinned to, or None."""
    acct = getattr(g, "account", None)
    return acct["pbx"] if acct else None


def _body() -> dict:
    return request.get_json(force=True, silent=True) or {}


# ---------------------------------------------------------------------------
# PBX (admin only)
# ---------------------------------------------------------------------------

@api.route("/pbx/<pbx_id>", methods=["PUT"])
@auth.require_admin
def put_pbx(pbx_id: str):
    body = _body()
    entry = pbx.put(pbx_id, body)
    # pbx.put() auto-registers trunk with sip_stack for outbound PSTN calls
    pbx_data = pbx.get(pbx_id)
    from . import sip_listener
    try:
        sip_listener.start_listener(pbx_id, pbx_data)
    except Exception as e:
        _LOGGER.warning("SIP listener start failed for %s: %s", pbx_id, e)
    return jsonify(entry), 200


@api.route("/pbx", methods=["GET"])
@auth.require_admin
def list_pbx():
    return jsonify(pbx.list_all())


@api.route("/pbx/<pbx_id>", methods=["DELETE"])
@auth.require_admin
def delete_pbx(pbx_id: str):
    if not pbx.delete(pbx_id):
        return ("PBX not found\n", 404)
    return ("", 204)


# ---------------------------------------------------------------------------
# Accounts (admin only)
# ---------------------------------------------------------------------------

@api.route("/accounts/<account_id>", methods=["PUT"])
@auth.require_admin
def put_account(account_id: str):
    try:
        acct = auth.put_account(account_id, _body())
    except ValueError as e:
        return (str(e) + "\n", 400)
    return jsonify(acct), 200


@api.route("/accounts", methods=["GET"])
@auth.require_admin
def list_accounts():
    return jsonify(auth.list_accounts())


@api.route("/accounts/<account_id>", methods=["GET"])
@auth.require_admin
def get_account(account_id: str):
    acct = auth.get_account(account_id)
    if not acct:
        return ("Account not found\n", 404)
    return jsonify(acct)


@api.route("/accounts/<account_id>", methods=["DELETE"])
@auth.require_admin
def delete_account(account_id: str):
    subscriber.delete_all_for_account(account_id)
    if not auth.delete_account(account_id):
        return ("Account not found\n", 404)
    return ("", 204)


# ---------------------------------------------------------------------------
# Subscribers (account-scoped)
# ---------------------------------------------------------------------------

@api.route("/subscribe/<subscriber_id>", methods=["PUT"])
@auth.require_account
def put_subscriber(subscriber_id: str):
    aid = _account_id()
    # Resolve PBX from account pin
    if aid:
        acct = auth.get_account(aid)
        if acct and acct.get("pbx"):
            if not pbx.get(acct["pbx"]):
                return (f"Account pinned to unknown PBX: {acct['pbx']}\n", 400)
    try:
        entry = subscriber.put(subscriber_id, aid or "__admin__", _body())
    except (PermissionError, ValueError) as e:
        return (str(e) + "\n", 403 if isinstance(e, PermissionError) else 400)
    return jsonify(entry), 200


@api.route("/subscribers", methods=["GET"])
@auth.require_account
def list_subscribers():
    return jsonify(subscriber.list_all(account_id=_account_id()))


@api.route("/subscribers/<subscriber_id>", methods=["GET"])
@auth.require_account
def get_subscriber(subscriber_id: str):
    entry = subscriber.get(subscriber_id)
    if not entry:
        return ("Subscriber not found\n", 404)
    aid = _account_id()
    if aid and entry["account_id"] != aid:
        return ("Forbidden\n", 403)
    return jsonify(entry)


@api.route("/subscribers/<subscriber_id>", methods=["DELETE"])
@auth.require_account
def delete_subscriber(subscriber_id: str):
    try:
        if not subscriber.delete(subscriber_id, account_id=_account_id()):
            return ("Subscriber not found\n", 404)
    except PermissionError as e:
        return (str(e) + "\n", 403)
    return ("", 204)


# ---------------------------------------------------------------------------
# Calls (account-scoped)
# ---------------------------------------------------------------------------

@api.route("/calls", methods=["POST"])
@auth.require_account
def create_call():
    """Create a new call (conference).

    Body: {subscriber_id, caller?, callee?, direction?, events?}
    Returns the call object including the generated callId.
    """
    body = _body()
    sub_id = body.get("subscriber_id", "")
    sub = subscriber.get(sub_id)
    if not sub:
        return ("Subscriber not found\n", 404)
    aid = _account_id()
    if aid and sub["account_id"] != aid:
        return ("Forbidden\n", 403)

    # Resolve PBX — account pin takes precedence
    acct = auth.get_account(sub["account_id"]) if sub["account_id"] != "__admin__" else None
    requested_pbx = body.get("pbx", "")
    if acct and acct.get("pbx"):
        pbx_id = acct["pbx"]
        if requested_pbx and requested_pbx != pbx_id:
            return (f"Account pinned to PBX {pbx_id}, cannot use {requested_pbx}\n", 403)
    else:
        pbx_id = requested_pbx

    call = call_state.create_call(
        subscriber_id=sub_id,
        account_id=sub["account_id"],
        pbx_id=pbx_id,
        caller=body.get("caller", ""),
        callee=body.get("callee", ""),
        direction=body.get("direction", "inbound"),
        events=sub.get("events", {}),
    )
    return jsonify(call.to_dict()), 201


@api.route("/calls", methods=["GET"])
@auth.require_account
def list_calls():
    calls = call_state.list_calls(account_id=_account_id())
    return jsonify([c.to_dict() for c in calls])


@api.route("/calls/<call_id>", methods=["GET"])
@auth.require_account
def get_call(call_id: str):
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)
    return jsonify(call.to_dict())


@api.route("/calls/<call_id>/participants", methods=["GET"])
@auth.require_account
def list_participants(call_id: str):
    """List active participants of a call (conference)."""
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)
    return jsonify(call.list_participants())


@api.route("/calls/<call_id>/commands", methods=["POST"])
@auth.require_account
def post_commands(call_id: str):
    """Send commands to a call.

    Body: {"commands": [{"action": "...", ...}, ...]}
    """
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)

    body = _body()
    commands = body.get("commands", [])
    if not isinstance(commands, list):
        return ("'commands' must be a list\n", 400)

    # Execute commands in background thread
    from . import commands as cmd_engine
    # Execute commands in background thread — return 202 immediately
    threading.Thread(
        target=cmd_engine.execute_commands,
        args=(call, commands),
        daemon=True,
        name=f"cmds-{call_id}",
    ).start()
    _LOGGER.info("Executing %d commands for call %s (background)", len(commands), call_id)

    return jsonify({"queued": len(commands), "call_id": call_id}), 202


@api.route("/calls/<call_id>/pipes", methods=["POST"])
@auth.require_account
def post_pipes(call_id: str):
    """Execute DSL pipes on a call.

    Body: {"pipes": ["sip:leg-abc -> call:xyz -> sip:leg-abc", ...]}
    """
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)

    body = _body()
    pipes = body.get("pipes", [])
    if not isinstance(pipes, list):
        return ("'pipes' must be a list\n", 400)

    from .pipe_executor import CallPipeExecutor
    if not hasattr(call, 'pipe_executor') or call.pipe_executor is None:
        sub = subscriber.get(call.subscriber_id) if hasattr(call, 'subscriber_id') else None
        from . import _shared
        call.pipe_executor = CallPipeExecutor(
            call, tts_registry=_shared.tts_registry, subscriber=sub)

    results = call.pipe_executor.add_pipes(pipes)
    return jsonify({"call_id": call_id, "results": results}), 202


@api.route("/calls/<call_id>/stages/<stage_id>", methods=["DELETE"])
@auth.require_account
def delete_stage(call_id: str, stage_id: str):
    """Kill a named stage (e.g., stop hold music)."""
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)
    if not hasattr(call, 'pipe_executor') or not call.pipe_executor:
        return ("No pipe executor\n", 400)
    if not call.pipe_executor.kill_stage(stage_id):
        return ("Stage not found\n", 404)
    return ("", 204)


@api.route("/calls/<call_id>", methods=["DELETE"])
@auth.require_account
def delete_call(call_id: str):
    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)

    # Hangup ALL legs for this call (BYE/CANCEL, Twilio semantics)
    from . import leg as leg_mod
    all_legs = list(leg_mod.list_legs())
    _LOGGER.info("delete_call %s: found %d legs, checking call_id match",
                 call_id, len(all_legs))
    for leg in all_legs:
        _LOGGER.info("  leg %s: call_id=%s status=%s has_sip_call=%s",
                     leg.leg_id, leg.call_id, leg.status,
                     hasattr(leg, '_sip_call') and leg._sip_call is not None)
        if leg.call_id == call_id and leg.status != "completed":
            _LOGGER.info("  -> deleting leg %s", leg.leg_id)
            leg.status = "completed"
            leg_mod.delete_leg(leg.leg_id)

    call.status = "completed"
    call_state.delete_call(call_id)
    return ("", 204)


# ---------------------------------------------------------------------------
# Legs (account-scoped)
# ---------------------------------------------------------------------------

@api.route("/legs", methods=["GET"])
@auth.require_account
def list_legs():
    """List SIP legs for this account's subscribers."""
    from . import leg as leg_mod
    aid = _account_id()
    # Filter by account's subscribers
    subs = subscriber.list_all(account_id=aid)
    sub_ids = {s["id"] for s in subs}
    legs = [l.to_dict() for l in leg_mod.list_legs()
            if l.subscriber_id in sub_ids or not aid]
    return jsonify(legs)


@api.route("/legs/<leg_id>", methods=["GET"])
@auth.require_account
def get_leg(leg_id: str):
    from . import leg as leg_mod
    l = leg_mod.get_leg(leg_id)
    if not l:
        return ("Leg not found\n", 404)
    return jsonify(l.to_dict())


@api.route("/legs/<leg_id>", methods=["DELETE"])
@auth.require_account
def delete_leg(leg_id: str):
    """Hang up a SIP leg."""
    from . import leg as leg_mod
    if not leg_mod.delete_leg(leg_id):
        return ("Leg not found\n", 404)
    return ("", 204)


@api.route("/legs/<leg_id>/answer", methods=["POST"])
@auth.require_account
def answer_leg(leg_id: str):
    """Answer an inbound leg (send 200 OK after Early Media).

    Call this when a participant has joined and the call should
    be fully established (stops Early Media, starts billing).
    """
    from . import leg as leg_mod, sip_stack

    l = leg_mod.get_leg(leg_id)
    if not l:
        return ("Leg not found\n", 404)

    # For sip_stack trunk legs: send 200 OK (Early Media → Answered)
    sip_call_id = getattr(l, '_sip_call_id', '')
    if sip_call_id:
        if sip_stack.answer_trunk_leg(sip_call_id):
            _LOGGER.info("Leg %s answered (200 OK sent)", leg_id)
            return jsonify({"answered": True, "leg_id": leg_id}), 200

    # For pyVoIP legs: already answered (pyVoIP limitation)
    if l.voip_call:
        return jsonify({"answered": True, "leg_id": leg_id, "note": "already answered (pyVoIP)"}), 200

    return ("Cannot answer this leg\n", 400)


@api.route("/legs/<leg_id>/bridge", methods=["POST"])
@auth.require_account
def bridge_leg(leg_id: str):
    """Bridge an existing leg into a conference.

    Body: {"call_id": "call-xxx", "callbacks": {"completed": "/path", ...}}
    Creates the call if call_id is not given.
    """
    from . import leg as leg_mod

    l = leg_mod.get_leg(leg_id)
    if not l:
        return ("Leg not found\n", 404)

    body = _body()
    call_id = body.get("call_id")
    callbacks = body.get("callbacks", {})
    l.callbacks.update(callbacks)

    if call_id:
        call = call_state.get_call(call_id)
        if not call:
            return ("Call not found\n", 404)
    else:
        # Auto-create a conference
        sub = subscriber.get(l.subscriber_id)
        call = call_state.create_call(
            subscriber_id=l.subscriber_id,
            account_id=sub["account_id"] if sub else "__admin__",
            pbx_id=l.pbx_id,
            caller=l.number if l.direction == "inbound" else "",
            callee=l.number if l.direction == "outbound" else "",
            direction=l.direction,
            events=sub.get("events", {}) if sub else {},
        )

    # Bridge via pipe_executor (DSL: sip:LEG -> call:CALL -> sip:LEG)
    from .pipe_executor import CallPipeExecutor
    if not hasattr(call, 'pipe_executor') or call.pipe_executor is None:
        sub = subscriber.get(call.subscriber_id) if hasattr(call, 'subscriber_id') else None
        from . import _shared
        call.pipe_executor = CallPipeExecutor(
            call, tts_registry=_shared.tts_registry, subscriber=sub)
    call.pipe_executor.add_pipes([
        f"sip:{leg_id} -> call:{call.call_id} -> sip:{leg_id}"
    ])
    return jsonify({"call_id": call.call_id, "leg_id": leg_id}), 200


@api.route("/legs/originate", methods=["POST"])
@auth.require_account
def originate_leg():
    """Originate an outbound SIP leg into an existing conference.

    Body: {"call_id": "call-xxx", "to": "+49170...",
           "callbacks": {"ringing": "/path", "answered": "/path",
                         "completed": "/path"}}
    """
    from . import leg as leg_mod, pbx as pbx_reg
    from . import commands as cmd_engine

    body = _body()
    call_id = body.get("call_id", "")
    to = body.get("to", "")
    callbacks = body.get("callbacks", {})

    call = call_state.get_call(call_id)
    if not call:
        return ("Call not found\n", 404)
    aid = _account_id()
    if aid and call.account_id != aid:
        return ("Forbidden\n", 403)
    if not to:
        return ("'to' is required\n", 400)
    # Enforce PBX pinning
    if aid and not auth.check_pbx_access(aid, call.pbx_id):
        return (f"Account not allowed to use PBX {call.pbx_id}\n", 403)

    pbx_entry = pbx_reg.get(call.pbx_id)
    if not pbx_entry:
        return (f"PBX {call.pbx_id} not found\n", 400)

    if to.startswith("client:"):
        from . import webclient as webclient_mod
        import secrets

        user = to[len("client:"):]
        if not user:
            return ("Invalid client target\n", 400)
        leg_id = "leg-" + secrets.token_urlsafe(12)
        ready_callback = body.get("webclient_callback", "")
        if not ready_callback:
            return ("'webclient_callback' is required for client targets\n", 400)
        base_url = body.get("base_url", "").rstrip("/")
        if not base_url:
            return ("'base_url' is required for client targets\n", 400)

        webclient_mod.create_webclient_leg(
            call=call,
            user=user,
            leg_id=leg_id,
            base_url=base_url,
            ready_callback=ready_callback,
            leg_callbacks=callbacks,
            number=to,
        )
        return jsonify({"leg_id": leg_id, "call_id": call_id}), 202

    leg = leg_mod.create_leg("outbound", to, call.pbx_id,
                              call.subscriber_id)
    leg.callbacks = callbacks
    leg.call_id = call_id  # set early so delete_call can find ringing legs
    leg.caller_id = body.get("caller_id")  # display name for remote party

    # Originate in background — fires ringing/answered/failed callbacks.
    # CRM bridges the leg via POST /pipes after receiving answered callback.
    threading.Thread(
        target=leg_mod.originate_only,
        args=(leg, pbx_entry),
        daemon=True,
        name=f"orig-{leg.leg_id}",
    ).start()

    return jsonify({"leg_id": leg.leg_id, "call_id": call_id}), 202


# ---------------------------------------------------------------------------
# Nonces (account-scoped)
# ---------------------------------------------------------------------------

@api.route("/nonce", methods=["POST"])
@auth.require_account
def create_nonce():
    body = _body()
    sub_id = body.get("subscriber_id", body.get("subscriber", ""))
    user = body.get("user", "")
    ttl = body.get("ttl", 3600)
    if not sub_id or not user:
        return ("'subscriber_id' and 'user' are required\n", 400)
    sub = subscriber.get(sub_id)
    if not sub:
        return ("Subscriber not found\n", 404)
    aid = _account_id()
    if aid and sub["account_id"] != aid:
        return ("Forbidden\n", 403)
    entry = auth.create_nonce(sub["account_id"], sub_id, user, ttl=ttl)
    return jsonify(entry), 201


@api.route("/nonces", methods=["GET"])
@auth.require_account
def list_nonces():
    aid = _account_id()
    return jsonify(auth.list_nonces(account_id=aid))


@api.route("/nonce/<nonce>", methods=["DELETE"])
@auth.require_account
def delete_nonce(nonce: str):
    if not auth.revoke_nonce(nonce):
        return ("Nonce not found\n", 404)
    return ("", 204)
