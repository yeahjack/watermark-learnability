import argparse
from datasets import load_dataset
import json
import os


def download_dataset(name, split, start, end, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 更改文件扩展名为 .txt
    output_file = os.path.join(
        output_dir, f"{name.split('/')[1]}_{split}_{end-start+1}.txt")

    dataset = load_dataset(name, split=split, streaming=True)
    samples = iter(dataset.skip(start - 1).take(end - start + 1))

    with open(output_file, 'w', encoding='utf-8') as file:
        for sample in samples:
            # 假设我们想将字典的每个键值对转换为一行文本
            for key, value in sample.items():
                # file.write(f"{key}: {value}\n")
                file.write(f"{value}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a subset of the C4 dataset.")
    parser.add_argument("--name",
                        type=str,
                        default="Skylion007/openwebtext",
                        help="The name of the dataset.")
    parser.add_argument("--split",
                        type=str,
                        default="train",
                        help="The dataset split to download.")
    parser.add_argument("--start",
                        type=int,
                        default=200001,
                        help="The start index of samples to download.")
    parser.add_argument("--end",
                        type=int,
                        default=300000,
                        help="The end index of samples to download.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=".",
                        help="The dir to the output file.")

    args = parser.parse_args()

    download_dataset(args.name, args.split, args.start, args.end,
                     args.output_dir)
