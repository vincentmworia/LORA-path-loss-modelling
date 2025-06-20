data = {
  "name": "as.up.data.forward",
  "time": "2025-05-28T19:34:14.206425Z",
  "identifiers": [
    {
      "device_ids": {
        "device_id": "pilotdevice02",
        "application_ids": {
          "application_id": "pilot-test"
        },
        "dev_eui": "A8610A3230378316",
        "join_eui": "0000000000000000",
        "dev_addr": "260BED96"
      }
    }
  ],
  "data": {
    "@type": "type.googleapis.com/ttn.lorawan.v3.ApplicationUp",
    "end_device_ids": {
      "device_id": "pilotdevice02",
      "application_ids": {
        "application_id": "pilot-test"
      },
      "dev_eui": "A8610A3230378316",
      "join_eui": "0000000000000000",
      "dev_addr": "260BED96"
    },
    "correlation_ids": [
      "gs:uplink:01JWC66PQE3XY7H2H21FASJMW9"
    ],
    "received_at": "2025-05-28T19:34:14.203818759Z",
    "uplink_message": {
      "session_key_id": "AZK+ev1SjvHPQqPySZa2Ew==",
      "f_port": 3,
      "f_cnt": 280418,
      "frm_payload": "fqUCFQlyEBgAPQAEvycAALcy",
      "decoded_payload": {
        "co2": 533,
        "humidity": 41.2,
        "packetCount": 311079,
        "pm25": 0.61,
        "pressure": 324.21,
        "temperature": 24.18
      },
      "rx_metadata": [
        {
          "gateway_ids": {
            "gateway_id": "kerlink001",
            "eui": "7276FF0039090946"
          },
          "time": "2025-05-28T19:34:13.955830Z",
          "timestamp": 2965558316,
          "rssi": -70,
          "channel_rssi": -70,
          "snr": 8,
          "uplink_token": "ChgKFgoKa2VybGluazAwMRIIcnb/ADkJCUYQrKiLhgsaDAi1yt3BBhD65sHbAyDgl7PIp+qgAQ==",
          "channel_index": 2,
          "received_at": "2025-05-28T19:34:11.718601156Z"
        },
        {
          "gateway_ids": {
            "gateway_id": "eui-a84041ffff22def8",
            "eui": "A84041FFFF22DEF8"
          },
          "timestamp": 2629044847,
          "rssi": -119,
          "channel_rssi": -119,
          "snr": -7.8,
          "frequency_offset": "-105",
          "uplink_token": "CiIKIAoUZXVpLWE4NDA0MWZmZmYyMmRlZjgSCKhAQf//It74EO+U0OUJGgsItsrdwQYQgMWSAyCYg6P6wd2hAQ==",
          "channel_index": 5,
          "received_at": "2025-05-28T19:34:13.988488840Z"
        }
      ],
      "settings": {
        "data_rate": {
          "lora": {
            "bandwidth": 125000,
            "spreading_factor": 11,
            "coding_rate": "4/5"
          }
        },
        "frequency": "867500000",
        "timestamp": 2965558316,
        "time": "2025-05-28T19:34:13.955830Z"
      },
      "received_at": "2025-05-28T19:34:13.999572325Z",
      "confirmed": true,
      "consumed_airtime": "0.987136s",
      "version_ids": {
        "brand_id": "arduino",
        "model_id": "mkr-wan-1310",
        "hardware_version": "1.0",
        "firmware_version": "1.2.3",
        "band_id": "EU_863_870"
      },
      "network_ids": {
        "net_id": "000013",
        "ns_id": "EC656E0000000181",
        "tenant_id": "ttn",
        "cluster_id": "eu1",
        "cluster_address": "eu1.cloud.thethings.network"
      }
    }
  },
  "correlation_ids": [
    "gs:uplink:01JWC66PQE3XY7H2H21FASJMW9"
  ],
  "origin": "ip-10-100-7-253.eu-west-1.compute.internal",
  "context": {
    "tenant-id": "CgN0dG4="
  },
  "visibility": {
    "rights": [
      "RIGHT_APPLICATION_TRAFFIC_READ"
    ]
  },
  "unique_id": "01JWC66PXY4NCRQ68736ZBWZN0"
}