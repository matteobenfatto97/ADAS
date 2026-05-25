from __future__ import annotations


def main() -> None:
    print("=== Torch / CUDA ===")
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        print("cuda devices:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("cuda device 0:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("torch check failed:", exc)

    print("\n=== OpenVINO ===")
    try:
        from openvino import Core
        core = Core()
        print("openvino devices:", core.available_devices)
        if "GPU" not in core.available_devices:
            print("[INFO] OpenVINO non vede GPU. Aggiorna driver Intel Graphics oppure usa intel:cpu.")
    except Exception as exc:
        print("openvino check failed:", exc)
        print("Installa: python -m pip install openvino")


if __name__ == "__main__":
    main()
