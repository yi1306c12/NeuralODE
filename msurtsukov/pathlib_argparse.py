if __name__ == "__main__":
    import argparse
    import pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(str(args.save_path))
    print(str(args.save_path.resolve()))