"""In-process CRM webhook receiver.

Mirrors the logic in ``fop-dev/backends/businesslogic/telefonanlage/
speech-server/webhooks.fop`` + ``outgoing.fop`` + ``calls.fop``.  Every
HTTP call the real CRM issues to the speech-server is reproduced here.

The server sends webhooks via ``_shared.post_webhook`` /
``dispatcher.http_requests.request`` (outbound HTTP).  Tests patch both
to route into ``FakeCrm.dispatch`` synchronously, and the CRM's own
outbound calls to the server go through the pytest Flask test client.

Usage
-----

    crm = FakeCrm(client, admin_headers=admin, account_token=tok)
    crm.register_as_subscriber(subscriber_id, pbx_id)
    with crm.active(monkeypatch):
        # trigger a flow that causes the server to fire webhooks
        ...
    # assert end state
    assert crm.calls[1]["status"] == "completed"
"""
from __future__ import annotations

import json
import re
import threading
import urllib.parse
from contextlib import contextmanager
from typing import Any


class FakeCrm:
    BASE_URL = "https://crm.example.test/crm"

    def __init__(self, client, *, admin_headers: dict, account_token: str):
        self.client = client
        self.admin_headers = admin_headers
        self.account_headers = {
            "Authorization": f"Bearer {account_token}",
            "Content-Type": "application/json",
        }
        self.account_token = account_token

        # Mini "DB" mirrored from webhooks.fop storage
        self.calls: dict[int, dict] = {}
        self.participants: dict[int, dict] = {}
        self.webhooks: list[dict] = []            # full audit trail
        self._next_call_id = 1
        self._next_pid = 1
        self._lock = threading.Lock()

        # Internal phones the CRM would dial when findParticipant runs.
        # Each entry: {"number": "+49...", "answerer_id": 42}
        self.internal_phones: list[dict] = []

        # Hold jingle URL (empty = no jingle).
        self.hold_jingle: str = ""
        self.wait_jingle: str = ""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_as_subscriber(self, subscriber_id: str, pbx_id: str) -> None:
        """Point the existing subscriber's base_url + events at us."""
        # Event keys must match what the server fires — see
        # ``sip_listener.fire_event(call, 'call_ended', ...)`` and
        # the real CRM's ``heartbeat.fop`` registration.
        events = {
            "incoming":    "/Telephone/SpeechServer/public?state=incoming",
            "device_dial": "/Telephone/SpeechServer/public?state=device-dial",
            "call_ended":  "POST /Telephone/SpeechServer/public?state=ended",
        }
        resp = self.client.put(
            f"/api/subscribe/{subscriber_id}",
            data=json.dumps({
                "base_url": self.BASE_URL,
                "bearer_token": self.account_token,
                "events": events,
            }),
            headers=self.account_headers,
        )
        assert resp.status_code in (200, 201), resp.data
        self._subscriber_cached = subscriber_id

    def _account_id_from_token(self) -> str:
        from speech_pipeline.telephony import auth as auth_mod
        for aid, acct in auth_mod._accounts.items():
            if acct["token"] == self.account_token:
                return aid
        raise RuntimeError("test account not registered")

    # ------------------------------------------------------------------
    # Webhook routing — monkeypatch entry point
    # ------------------------------------------------------------------

    @contextmanager
    def active(self, monkeypatch):
        """Install patches so outbound webhooks land in :meth:`dispatch`."""
        from speech_pipeline.telephony import _shared as sh

        def _fake_post_webhook(url, payload, bearer_token, **kw):
            self._route(url, payload, bearer_token, method="POST")

        def _fake_requests_request(method, url, json=None, **kwargs):
            self._route(url, json or {}, kwargs.get("headers", {}).get(
                "Authorization", "").removeprefix("Bearer "),
                method=method)
            return _FakeResponse(200, b'{"ok":true}')

        monkeypatch.setattr(sh, "post_webhook", _fake_post_webhook)
        import requests as _req
        monkeypatch.setattr(_req, "request", _fake_requests_request)
        monkeypatch.setattr(_req, "post",
                            lambda url, json=None, **kw:
                            _fake_requests_request("POST", url,
                                                   json=json, **kw))
        yield

    def _route(self, url: str, payload: dict, token: str,
               method: str = "POST") -> None:
        if not url.startswith(self.BASE_URL):
            return  # not for us
        parsed = urllib.parse.urlsplit(url)
        qs = urllib.parse.parse_qs(parsed.query)
        state = qs.get("state", [""])[0]
        self.webhooks.append({
            "state": state,
            "query": {k: v[0] for k, v in qs.items()},
            "body": payload,
            "method": method,
        })
        handler = getattr(self, f"_on_{state.replace('-', '_')}", None)
        if handler:
            handler(qs, payload)

    # ------------------------------------------------------------------
    # webhooks.fop::publicAction — one handler per ``state``
    # ------------------------------------------------------------------

    def _on_incoming(self, qs, body):
        """state=incoming: create call record + bridge + findParticipant."""
        caller = body.get("caller", "")
        callee = body.get("callee", "")
        leg_id = body.get("leg_id", "")

        # Create call row (mirrors Call::create)
        with self._lock:
            call_db_id = self._next_call_id
            self._next_call_id += 1
        self.calls[call_db_id] = {
            "caller": caller,
            "direction": "inbound",
            "status": "ringing",
            "sid": "",
        }

        # POST /api/calls
        resp = self.client.post(
            "/api/calls",
            data=json.dumps({
                "subscriber_id": self._subscriber_id(),
                "caller": caller,
                "callee": callee,
                "direction": "inbound",
            }),
            headers=self.account_headers,
        )
        if resp.status_code != 201:
            return
        call_sid = resp.get_json()["call_id"]
        self.calls[call_db_id]["sid"] = call_sid

        # Build and post the leg pipe (with STT tee) + wait music.
        cb_completed = (
            f"/Telephone/SpeechServer/public?state=leg&event=completed"
            f"&call={call_db_id}"
        )
        cb = {"completed": cb_completed}
        # buildLegPipes hook (transcript.fop adds STT tap).
        tap = f"{leg_id}_tap"
        sip_pipe = (
            f"sip:{leg_id}{json.dumps(cb)} -> tee:{tap} "
            f"-> call:{call_sid} -> sip:{leg_id}"
        )
        stt_pipe = (
            f"tee:{tap} -> stt:de -> webhook:"
            f"{self.BASE_URL}/Telephone/SpeechServer/sttNote"
            f"?call={call_db_id}&participant=0"
        )
        for pipe in (sip_pipe, stt_pipe):
            self.client.post("/api/pipelines",
                             data=json.dumps({"dsl": pipe}),
                             headers=self.account_headers)

        if self.wait_jingle:
            self.client.post("/api/pipelines", data=json.dumps({
                "dsl": f'play:{call_sid}_wait'
                       f'{{"url":"{self.wait_jingle}","loop":true,"volume":50}} '
                       f'-> call:{call_sid}',
            }), headers=self.account_headers)

        # findParticipant — dial each internal phone via originate DSL.
        self._find_participant(call_db_id)

    def _on_leg(self, qs, body):
        """state=leg with an &event=... query param."""
        event = qs.get("event", [""])[0]
        call_db_id = int(qs.get("call", [0])[0])
        pid = int(qs.get("participant", [0])[0])
        leg_sid = body.get("leg_id", "") or qs.get("leg_id", [""])[0]

        if event == "ringing":
            if pid in self.participants:
                self.participants[pid]["status"] = "ringing"
        elif event == "answered":
            self._on_leg_answered(call_db_id, pid, leg_sid)
        elif event in ("no-answer", "busy", "failed", "canceled"):
            self._on_leg_ended(call_db_id, pid, event)
        elif event == "completed":
            self._on_leg_ended(call_db_id, pid, "completed")

    def _on_webclient(self, qs, body):
        call_db_id = int(qs.get("call", [0])[0])
        pid = int(qs.get("participant", [0])[0])
        iframe = body.get("iframe_url", "")
        sess = body.get("session_id") or body.get("leg_id", "")
        if call_db_id and iframe:
            self.calls.setdefault(call_db_id, {})["webclient_url"] = iframe
        if pid and sess:
            self.participants.setdefault(pid, {})["sid"] = sess
            self.participants[pid]["number"] = "webclient"

    def _on_ended(self, qs, body):
        sid = body.get("callId", "")
        for cid, c in self.calls.items():
            if c.get("sid") == sid:
                c["status"] = "completed"

    def _on_device_dial(self, qs, body):
        """A registered SIP device dialed → create outbound call record +
        attach leg + originate external participant."""
        number = body.get("number") or body.get("callee", "")
        leg_id = body.get("leg_id", "")
        sip_user = body.get("sip_user", "")
        if not number or not leg_id:
            return

        with self._lock:
            call_db_id = self._next_call_id
            self._next_call_id += 1
            pid = self._next_pid
            self._next_pid += 1
        self.calls[call_db_id] = {
            "caller": sip_user,
            "direction": "outbound",
            "callee": number,
            "status": "ringing",
            "sid": "",
        }
        self.participants[pid] = {
            "call_db_id": call_db_id,
            "sid": leg_id,
            "status": "answered",
            "number": sip_user,
        }

        # ensureOutboundConference
        call_sid = self._ensure_outbound_conference(call_db_id, number)
        if not call_sid:
            return
        # attachExistingLegToOutboundCall
        self._attach_leg_to_outbound(call_db_id, pid, leg_id, sip_user, number)
        # originateOutboundExternal
        self._originate_outbound_external(call_db_id, number)

    # ------------------------------------------------------------------
    # CRM-side helpers (mirror the .fop helper functions)
    # ------------------------------------------------------------------

    def _subscriber_id(self) -> str:
        from speech_pipeline.telephony import auth as auth_mod
        # Return the subscriber id registered in this account.
        for _sid, s in __import__(
            "speech_pipeline.telephony.subscriber", fromlist=["*"]
        )._subscribers.items():
            if s.get("account_id") == self._account_id_from_token():
                return _sid
        raise RuntimeError("no subscriber registered")

    def _find_participant(self, call_db_id: int) -> None:
        """calls.fop::findParticipant — dial each internal phone."""
        call_sid = self.calls[call_db_id]["sid"]
        for phone in self.internal_phones:
            with self._lock:
                pid = self._next_pid
                self._next_pid += 1
            self.participants[pid] = {
                "call_db_id": call_db_id,
                "number": phone["number"],
                "status": "adding",
                "sid": "",
            }
            cb_base = (
                f"/Telephone/SpeechServer/public?call={call_db_id}"
                f"&participant={pid}"
            )
            cb = {
                "caller_id": "",
                "ringing":   cb_base + "&state=leg&event=ringing",
                "answered":  cb_base + "&state=leg&event=answered",
                "completed": cb_base + "&state=leg&event=completed",
                "failed":    cb_base + "&state=leg&event=failed",
                "no-answer": cb_base + "&state=leg&event=no-answer",
                "busy":      cb_base + "&state=leg&event=busy",
            }
            dsl = (
                f"originate:{phone['number']}{json.dumps(cb)} "
                f"-> call:{call_sid}"
            )
            resp = self.client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": dsl}),
                headers=self.account_headers,
            )
            if resp.status_code == 201 and resp.get_json().get("leg_id"):
                self.participants[pid]["sid"] = resp.get_json()["leg_id"]

    def _on_leg_answered(self, call_db_id: int, pid: int, leg_sid: str) -> None:
        """Bridge leg into conference, stop wait music."""
        call_data = self.calls.get(call_db_id)
        if not call_data or not call_data.get("sid") or not leg_sid:
            return
        call_sid = call_data["sid"]

        # Update status.
        if pid and pid in self.participants:
            self.participants[pid]["status"] = "answered"
        if call_data["status"] in ("ringing", "adding", "bouncing"):
            call_data["status"] = "answered"

        # Bridge the leg with STT tap.
        cb_completed = (
            f"/Telephone/SpeechServer/public?call={call_db_id}"
            f"&participant={pid}&state=leg&event=completed"
        )
        tap = f"{leg_sid}_tap"
        sip_pipe = (
            f"sip:{leg_sid}{json.dumps({'completed': cb_completed})} "
            f"-> tee:{tap} -> call:{call_sid} -> sip:{leg_sid}"
        )
        stt_pipe = (
            f"tee:{tap} -> stt:de -> webhook:"
            f"{self.BASE_URL}/Telephone/SpeechServer/sttNote"
            f"?call={call_db_id}&participant={pid}"
        )
        for pipe in (sip_pipe, stt_pipe):
            self.client.post("/api/pipelines",
                             data=json.dumps({"dsl": pipe}),
                             headers=self.account_headers)

        # Outbound: keep wait music until internal answers; inbound:
        # stop wait music, auto-answer inbound SIP participants.
        keep_wait = bool(pid) and call_data.get("direction") == "outbound"
        if not keep_wait:
            self.client.delete(
                "/api/pipelines",
                data=json.dumps({"dsl": f"play:{call_sid}_wait"}),
                headers=self.account_headers,
            )
        if call_data.get("direction") == "inbound":
            info = self.client.get(
                f"/api/pipelines?dsl=call:{call_sid}",
                headers=self.account_headers,
            )
            if info.status_code == 200:
                for p in info.get_json().get("participants", []):
                    if p.get("type") == "sip" and p.get("direction") == "inbound":
                        self.client.post(
                            "/api/pipelines",
                            data=json.dumps({"dsl": f"answer:{p['id']}"}),
                            headers=self.account_headers,
                        )

    def _on_leg_ended(self, call_db_id: int, pid: int, reason: str) -> None:
        if pid and pid in self.participants:
            self.participants[pid]["status"] = reason
            self.participants[pid]["end_reason"] = reason
        if call_db_id not in self.calls:
            return
        call = self.calls[call_db_id]
        remaining = [
            (p_id, p) for p_id, p in self.participants.items()
            if p.get("call_db_id") == call_db_id
            and p.get("status") in ("answered", "ringing", "adding",
                                     "bouncing", "hold")
        ]
        active = [p for _, p in remaining if p.get("status") == "answered"]
        # Liveliness: if no active participants remain, end the call.
        if not remaining:
            call["status"] = "completed"
            call_sid = call.get("sid")
            if call_sid:
                self.client.delete(f"/api/calls/{call_sid}",
                                    headers=self.account_headers)
            return
        # Auto-unhold: last non-hold left while someone else is on hold
        # → revert call status + unhold.  Mirrors
        # ``calls.fop::checkLiveliness`` (speech-server-hold hook).
        if (len(remaining) == 1
                and call.get("status") in ("hold", "hold-adding")
                and not active):
            p_id, p = remaining[0]
            call["status"] = "answered"
            self.unhold_external_legs(call_db_id, p_id, p.get("sid", ""))

    def _ensure_outbound_conference(self, call_db_id: int, number: str) -> str:
        call = self.calls.get(call_db_id, {})
        if call.get("sid"):
            return call["sid"]
        resp = self.client.post(
            "/api/calls",
            data=json.dumps({
                "subscriber_id": self._subscriber_id(),
                "caller": "",
                "callee": number,
                "direction": "outbound",
            }),
            headers=self.account_headers,
        )
        if resp.status_code != 201:
            return ""
        call_sid = resp.get_json()["call_id"]
        self.calls[call_db_id]["sid"] = call_sid
        if self.wait_jingle:
            self.client.post("/api/pipelines", data=json.dumps({
                "dsl": f'play:{call_sid}_wait'
                       f'{{"url":"{self.wait_jingle}","loop":true,"volume":50}} '
                       f'-> call:{call_sid}',
            }), headers=self.account_headers)
        return call_sid

    def _attach_leg_to_outbound(self, call_db_id, pid, leg_id,
                                 sip_user, number):
        call = self.calls.get(call_db_id)
        if not call or not call.get("sid"):
            return
        call_sid = call["sid"]
        cb_completed = (
            f"/Telephone/SpeechServer/public?call={call_db_id}"
            f"&participant={pid}&state=leg&event=completed"
        )
        tap = f"{leg_id}_tap"
        sip_pipe = (
            f"sip:{leg_id}{json.dumps({'completed': cb_completed})} "
            f"-> tee:{tap} -> call:{call_sid} -> sip:{leg_id}"
        )
        stt_pipe = (
            f"tee:{tap} -> stt:de -> webhook:"
            f"{self.BASE_URL}/Telephone/SpeechServer/sttNote"
            f"?call={call_db_id}&participant={pid}"
        )
        for pipe in (sip_pipe, stt_pipe):
            self.client.post("/api/pipelines",
                             data=json.dumps({"dsl": pipe}),
                             headers=self.account_headers)
        self.client.post(
            "/api/pipelines",
            data=json.dumps({"dsl": f"answer:{leg_id}"}),
            headers=self.account_headers,
        )

    def _originate_outbound_external(self, call_db_id: int, number: str):
        call = self.calls.get(call_db_id)
        if not call or not call.get("sid"):
            return
        call_sid = call["sid"]
        with self._lock:
            pid = self._next_pid
            self._next_pid += 1
        self.participants[pid] = {
            "call_db_id": call_db_id,
            "number": number,
            "status": "adding",
            "sid": "",
        }
        cb_base = f"/Telephone/SpeechServer/public?state=leg&call={call_db_id}"
        cb = {
            "caller_id": "",
            "ringing":   cb_base + "&event=ringing",
            "answered":  cb_base + "&event=answered",
            "completed": cb_base + "&event=completed",
            "failed":    cb_base + "&event=failed",
            "no-answer": cb_base + "&event=no-answer",
            "busy":      cb_base + "&event=busy",
            "canceled":  cb_base + "&event=canceled",
        }
        dsl = (
            f"originate:{number}{json.dumps(cb)} -> call:{call_sid}"
        )
        resp = self.client.post(
            "/api/pipelines",
            data=json.dumps({"dsl": dsl}),
            headers=self.account_headers,
        )
        if resp.status_code == 201:
            self.participants[pid]["sid"] = resp.get_json().get("leg_id", "")

    # ------------------------------------------------------------------
    # Hold / Unhold — mirrors calls.fop::{hold,unhold}ExternalLegs
    # ------------------------------------------------------------------

    def hold_external_legs(self, call_db_id: int, pid: int,
                           leg_id: str) -> None:
        """CRM holdExternalLegs: drop the conference bridge, play the
        hold jingle straight into the leg."""
        call = self.calls.get(call_db_id)
        if not call or not call.get("sid"):
            return
        call_sid = call["sid"]
        # stopWaitMusic.
        self.client.delete("/api/pipelines",
                            data=json.dumps({"dsl": f"play:{call_sid}_wait"}),
                            headers=self.account_headers)
        # Drop bridge.
        self.client.delete("/api/pipelines",
                            data=json.dumps({"dsl": f"bridge:{leg_id}"}),
                            headers=self.account_headers)
        # Idempotent DELETE of any prior hold-music stage.
        stage = f"play:{call_sid}_hold_{leg_id}"
        self.client.delete("/api/pipelines",
                            data=json.dumps({"dsl": stage}),
                            headers=self.account_headers)
        if self.hold_jingle:
            self.client.post("/api/pipelines", data=json.dumps({
                "dsl": f'{stage}'
                       f'{{"url":"{self.hold_jingle}","loop":true,"volume":50}} '
                       f'-> sip:{leg_id}',
            }), headers=self.account_headers)
        if pid and pid in self.participants:
            self.participants[pid]["status"] = "hold"
        call["status"] = "hold"

    def unhold_external_legs(self, call_db_id: int, pid: int,
                             leg_id: str) -> None:
        """CRM unholdExternalLegs: stop hold music, rebuild the bridge
        — with the ``completed`` callback correctly carrying both the
        call DB id and the participant id, so later teardowns reach the
        right CRM row."""
        call = self.calls.get(call_db_id)
        if not call or not call.get("sid"):
            return
        call_sid = call["sid"]
        # Stop hold music.
        stage = f"play:{call_sid}_hold_{leg_id}"
        self.client.delete("/api/pipelines",
                            data=json.dumps({"dsl": stage}),
                            headers=self.account_headers)
        # Rebuild bridge with completed callback — BOTH call + participant
        # in the URL so webhook recipients don't have to guess.
        cb_completed = (
            f"/Telephone/SpeechServer/public?call={call_db_id}"
            f"&participant={pid}&state=leg&event=completed"
        )
        dsl = (f"sip:{leg_id}{json.dumps({'completed': cb_completed})} "
               f"-> call:{call_sid} -> sip:{leg_id}")
        self.client.post("/api/pipelines",
                         data=json.dumps({"dsl": dsl}),
                         headers=self.account_headers)
        if pid and pid in self.participants:
            self.participants[pid]["status"] = "answered"
        call["status"] = "answered"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_active_call_db_id(self) -> int | None:
        for cid, c in self.calls.items():
            if c.get("status") in ("ringing", "adding", "answered",
                                    "hold-adding", "hold"):
                return cid
        return None


class _FakeResponse:
    def __init__(self, code: int, content: bytes):
        self.status_code = code
        self.content = content
        self.text = content.decode(errors="replace")

    def json(self):
        return json.loads(self.text) if self.text else {}
