import base64
import os
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class Ed25519Signer:
    """
    Out-of-Band Ed25519 Cryptographic Signer:
    Uses root private key provisioned outside the main application package.
    Configurable via constructor argument or RONOVA_PRIVATE_KEY_PATH environment variable.
    Fails closed without automatic key generation during API operation.
    """

    def __init__(self, private_key_path: Optional[str] = None):
        self._explicit_private_key_path = private_key_path

    @property
    def private_key_path(self) -> Path:
        env_path = os.environ.get("RONOVA_PRIVATE_KEY_PATH")
        path = self._explicit_private_key_path or env_path or "secrets/root_private.pem"
        return Path(path).resolve()

    def generate_keypair(
        self,
        public_key_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Tuple[str, str]:
        """
        Explicit administrative operation to generate out-of-band Ed25519 keypair.
        Will NOT overwrite existing keys unless overwrite=True.
        """
        priv_path = self.private_key_path

        env_pub = os.environ.get("RONOVA_PUBLIC_KEY_PATH")
        pub_str = public_key_path or env_pub or "trust/root_public.pem"
        pub_path = Path(pub_str).resolve()

        if priv_path.exists() and not overwrite:
            raise FileExistsError(
                f"Private key already exists at {priv_path}. Use overwrite=True to replace."
            )

        if pub_path.exists() and not overwrite:
            raise FileExistsError(
                f"Public key already exists at {pub_path}. Use overwrite=True to replace."
            )

        priv_key = ed25519.Ed25519PrivateKey.generate()
        pub_key = priv_key.public_key()

        os.makedirs(priv_path.parent, exist_ok=True)
        os.makedirs(pub_path.parent, exist_ok=True)

        # Write private key (PEM format)
        with open(priv_path, "wb") as f:
            f.write(
                priv_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write public key (PEM format)
        with open(pub_path, "wb") as f:
            f.write(
                pub_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

        return str(priv_path), str(pub_path)

    def sign_message(self, message_bytes: bytes) -> str:
        """Signs raw message bytes and returns Base64 encoded signature string."""
        priv_path = self.private_key_path
        if not priv_path or not priv_path.exists():
            raise FileNotFoundError(
                f"Out-of-band private key not found at {priv_path}. Run administrative key setup first."
            )

        try:
            with open(priv_path, "rb") as f:
                key_bytes = f.read()
                priv_key = serialization.load_pem_private_key(key_bytes, password=None)
        except Exception as e:
            raise ValueError(f"Invalid private key file at {priv_path}: {str(e)}")

        if not isinstance(priv_key, ed25519.Ed25519PrivateKey):
            raise TypeError(f"Loaded key at {priv_path} is not an Ed25519 private key")

        sig_bytes = priv_key.sign(message_bytes)
        return base64.b64encode(sig_bytes).decode("utf-8")


class Ed25519Verifier:
    """
    Offline Ed25519 Signature Verifier:
    Uses read-only public key provisioned in trust/root_public.pem or via RONOVA_PUBLIC_KEY_PATH.
    Operates independently without access to private key.
    """

    def __init__(self, public_key_path: Optional[str] = None):
        self._explicit_public_key_path = public_key_path

    @property
    def public_key_path(self) -> Path:
        env_path = os.environ.get("RONOVA_PUBLIC_KEY_PATH")
        path = self._explicit_public_key_path or env_path or "trust/root_public.pem"
        return Path(path).resolve()

    def verify_signature(self, message_bytes: bytes, b64_signature: str) -> bool:
        """Verifies message bytes against Base64 signature string using public key."""
        pub_path = self.public_key_path
        if not pub_path or not pub_path.exists():
            raise FileNotFoundError(f"Trust Root public key not found at {pub_path}")

        try:
            with open(pub_path, "rb") as f:
                key_bytes = f.read()
                pub_key = serialization.load_pem_public_key(key_bytes)
        except Exception as e:
            raise ValueError(f"Invalid public key file at {pub_path}: {str(e)}")

        if not isinstance(pub_key, ed25519.Ed25519PublicKey):
            raise TypeError(f"Loaded key at {pub_path} is not an Ed25519 public key")

        try:
            sig_bytes = base64.b64decode(b64_signature)
            pub_key.verify(sig_bytes, message_bytes)
            return True
        except Exception:
            return False
