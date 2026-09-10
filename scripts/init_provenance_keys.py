import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ronova.provenance.crypto import Ed25519Signer


def main():
    parser = argparse.ArgumentParser(description="RONOVA Administrative Provenance Key Generator")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing key files")
    parser.add_argument("--priv", type=str, default=None, help="Custom private key destination path")
    parser.add_argument("--pub", type=str, default=None, help="Custom public key destination path")
    args = parser.parse_args()

    signer = Ed25519Signer(private_key_path=args.priv)
    try:
        priv_path, pub_path = signer.generate_keypair(public_key_path=args.pub, overwrite=args.force)
        print("=======================================================")
        print("RONOVA ADMINISTRATIVE KEYPAIR INITIALIZATION")
        print("=======================================================")
        print(f"Out-of-band Private Key (Secrets) : {priv_path}")
        print(f"Read-Only Public Key (Trust Root) : {pub_path}")
        print("-------------------------------------------------------")
        print("IMPORTANT: Keep private key out of source control and application path.")
        print("=======================================================")
    except FileExistsError as e:
        print(f"ADMIN ERROR: {str(e)}")
        print("Use --force argument if you explicitly intend to overwrite existing keys.")
        sys.exit(1)


if __name__ == "__main__":
    main()
