from diffusers.utils import check_min_version

from ReVersion import ReVersionTrainer

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")


def main():
    trainer = ReVersionTrainer()
    trainer.forward()

if __name__ == "__main__":
    main()
