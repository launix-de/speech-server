"""Telephony integration module.

Provides PBX management, account/subscriber registration, and call
control via a command-based API.  All state is held in memory only —
on restart the ``--startup-callback`` re-provisions everything.
"""
