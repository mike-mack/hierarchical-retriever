{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.nodejs_20
    pkgs.python311
    pkgs.python3Packages.pip
    pkgs.git
  ];

  shellHook = ''
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    '';
}