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
- ``DELETE /api/calls/<call_id>``     – end a call
  (details/participants: GET /api/pipelines?dsl=call:<id>)
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

    # Enforce max_concurrent_calls
    acct = auth.get_account(sub["account_id"]) if sub["account_id"] != "__admin__" else None
    if acct:
        max_calls = acct.get("max_concurrent_calls", 0)
        if max_calls > 0:
            current = len(call_state.list_calls(account_id=sub["account_id"]))
            if current >= max_calls:
                return ("Max concurrent calls exceeded\n", 429)

    # Resolve PBX — account pin takes precedence
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


# Legacy GET /api/calls/<id> and /participants endpoints removed —
# use GET /api/pipelines?dsl=call:ID instead (returns participants too).


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
                     hasattr(leg, 'sip_call') and leg.sip_call is not None)
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

# Legacy /api/legs/* endpoints removed — use /api/pipelines DSL instead:
# - List/get leg: GET /api/pipelines?dsl=sip:LEG_ID
# - Delete leg: DELETE /api/pipelines {"dsl": "bridge:LEG_ID"} (kills bridge)
# - Answer leg: POST /api/pipelines {"dsl": "answer:LEG_ID"}
# - Bridge leg: POST /api/pipelines {"dsl": "sip:LEG -> call:CALL -> sip:LEG"}
# - Originate: POST /api/pipelines {"dsl": "originate:NUM{cb} -> call:CALL"}


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
