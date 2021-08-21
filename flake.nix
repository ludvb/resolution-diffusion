{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgName = "resolution-diffusion";
        pkgs = import nixpkgs {
          inherit system;
          overlays = import ./nix/overlays.nix {
            hostPkgs = import nixpkgs { inherit system; };
          };
        };
        python = (pkgs.python39.withPackages
          (ps: with ps; [ black ipython isort poetry ])).override {
            ignoreCollisions = true;
          };

      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ glibcLocales pyright python ];
          shellHook = ''
            # export PYTHONPATH=${python}/lib/python3.9/site-packages
            # export LD_PRELOAD+=:/usr/lib/libcuda.so
            # export LD_PRELOAD+=:/usr/lib/libnvidia-ptxjitcompiler.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_adv_infer.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_adv_train.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_cnn_infer.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_cnn_train.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_ops_infer.so
            # export LD_PRELOAD+=:/usr/lib/libcudnn_ops_train.so
            # export LD_LIBRARY_PATH+=:/opt/cuda/targets/x86_64-linux/lib:${pkgs.gcc-unwrapped.lib}/lib
          '';
        };
      });
}
