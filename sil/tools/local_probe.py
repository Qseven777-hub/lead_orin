#!/usr/bin/env python3
"""Host-side probe: send one SIL frame and print the control the peer returned.

Runs in the Python 3.10 ``cvci_project`` environment against a running bridge
and peer (the stub or the Orin). Used by ``sil/run_loopback.sh``.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lead.evaluation.sil import codec, contract
from lead.evaluation.sil.transport import SilTransport

TIMEOUT_MS = 5000


def main() -> int:
    transport = SilTransport()
    transport.send(
        contract.TOPIC_SESSION,
        codec.encode(
            contract.session(
                route_id="1711",
                session_id="probe",
                scenario_type="ParkingCutIn",
                map_name="Town12",
                gnss_uses_transverse_mercator=False,
                global_plan_gps=[{"lat": 1.0, "lon": 2.0, "z": 0.0, "command": 4}],
                lat_ref=1.0,
                lon_ref=2.0,
                camera_indices=(1, 2, 3),
                config_source="probe",
            ),
        ),
    )

    # The bridge publishes each frame once and nothing is latched, so resend
    # until the ROS graph has connected the peer and a control comes back.
    deadline = time.monotonic() + TIMEOUT_MS / 1000.0
    seq = 0
    while time.monotonic() < deadline:
        seq += 1
        transport.send(
            contract.TOPIC_SENSOR,
            codec.encode(
                contract.sensor_frame(
                    seq=seq,
                    step=seq,
                    sim_time_us=seq * 50_000,
                    sensors={
                        "lidar1": np.zeros((4, 3), dtype=np.float32),
                        "speed": np.float32(3.0),
                    },
                    camera_indices=(1, 2, 3),
                ),
            ),
        )
        reply = transport.wait_for(contract.TOPIC_CONTROL, timeout_ms=500)
        if reply is not None:
            break

    transport.close()
    if reply is None:
        print(f"FAIL: no control within {TIMEOUT_MS} ms")
        return 1
    control = codec.decode(reply)
    print(
        f"OK: control seq={control['seq']} steer={control['steer']} "
        f"throttle={control['throttle']} brake={control['brake']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
