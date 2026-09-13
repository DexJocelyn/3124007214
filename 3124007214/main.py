import sys
import re
from collections import Counter
import math


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk", errors="ignore") as f:
                return f.read()
        except OSError as e:
            print("警告：读取文件失败", path, e)
            return ""
    except OSError as e:
        print("警告：读取文件失败", path, e)
        return ""


def clean_text(text):
    return re.sub(r"[^\w]+", "", text)


def split_grams(text, n=2):
    grams = []
    for i in range(len(text) - n + 1):
        grams.append(text[i:i + n])
    return grams


def count_grams(text, n=2):
    return Counter(split_grams(clean_text(text), n))


def cosine_similarity(freq_a, freq_b):
    dot = 0
    for k, v in freq_a.items():
        dot += v * freq_b.get(k, 0)

    sum_sq_orig = 0
    for v in freq_a.values():
        sum_sq_orig += v * v
    norm_orig = math.sqrt(sum_sq_orig)

    sum_sq_copy = 0
    for v in freq_b.values():
        sum_sq_copy += v * v
    norm_copy = math.sqrt(sum_sq_copy)

    if norm_orig == 0 or norm_copy == 0:
        return 0.0

    return dot / (norm_orig * norm_copy)


def calc_similarity(text_a, text_b):
    return cosine_similarity(count_grams(text_a), count_grams(text_b))


def write_answer(path, similarity):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{similarity:.2f}")
        return True
    except OSError as e:
        print("警告：写入答案文件失败", path, e)
        return False


def main():
    if len(sys.argv) != 4:
        print("用法: python main.py <原文路径> <抄袭版路径> <答案输出路径>")
        return

    orig_path = sys.argv[1]
    copy_path = sys.argv[2]
    out_path = sys.argv[3]

    orig_text = read_text(orig_path)
    copy_text = read_text(copy_path)

    similarity = calc_similarity(orig_text, copy_text)
    print("重复率：", similarity)

    if write_answer(out_path, similarity):
        print("答案已写入文件:", out_path)


if __name__ == "__main__":  # pragma: no cover
    main()