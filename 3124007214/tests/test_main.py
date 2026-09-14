import sys
from collections import Counter
from unittest.mock import patch

import pytest

import main


ORIG = "今天是星期天，天气晴，今天晚上我要去看电影。"
COPY = "今天是周天，天气晴朗，我晚上要去看电影。"


@pytest.mark.parametrize(
    ("text_a", "text_b", "expected"),
    [
        (ORIG, ORIG, 1.0),
        (ORIG, COPY, 0.6149186938124421),
        ("", "", 0.0),
        ("", "任意内容", 0.0),
        ("今天，天气！", "今天天气", 1.0),
        ("abc", "xyz", 0.0),
    ],
)
def test_calc_similarity(text_a, text_b, expected):
    result = main.calc_similarity(text_a, text_b)
    assert result == pytest.approx(expected)


def test_count_grams():
    result = main.count_grams("ABCD", n=2)

    assert result["AB"] == 1
    assert result["BC"] == 1
    assert result["CD"] == 1


def test_write_answer(tmp_path):
    output_path = tmp_path / "ans.txt"

    assert main.write_answer(str(output_path), 0.614918)
    assert output_path.read_text(encoding="utf-8") == "0.61"


def test_read_utf8_file(tmp_path):
    input_path = tmp_path / "orig.txt"
    input_path.write_text("测试文本", encoding="utf-8")

    assert main.read_text(str(input_path)) == "测试文本"


def test_wrong_number_of_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    output = capsys.readouterr().out
    assert "用法:" in output


def test_clean_text():
    assert main.clean_text("今天，天气！ 很好。") == "今天天气很好"


def test_split_grams():
    assert main.split_grams("ABCD", n=2) == ["AB", "BC", "CD"]
    assert main.split_grams("A", n=2) == []


def test_cosine_similarity_zero_vector():
    assert main.cosine_similarity(Counter(), Counter({"AB": 1})) == 0.0
    assert main.cosine_similarity(Counter({"AB": 1}), Counter()) == 0.0


def test_read_missing_file(tmp_path, capsys):
    missing_path = tmp_path / "not_exist.txt"

    result = main.read_text(str(missing_path))

    assert result == ""
    assert "读取文件失败" in capsys.readouterr().out


def test_read_gbk_file(tmp_path):
    gbk_path = tmp_path / "gbk.txt"
    gbk_path.write_bytes("中文测试".encode("gbk"))

    assert main.read_text(str(gbk_path)) == "中文测试"


def test_write_answer_invalid_path(tmp_path, capsys):
    invalid_path = tmp_path / "not_exist_dir" / "ans.txt"

    result = main.write_answer(str(invalid_path), 0.5)

    assert result is False
    assert "写入答案文件失败" in capsys.readouterr().out


def test_main_success(monkeypatch, tmp_path, capsys):
    orig_path = tmp_path / "orig.txt"
    copy_path = tmp_path / "copy.txt"
    ans_path = tmp_path / "ans.txt"

    orig_path.write_text(ORIG, encoding="utf-8")
    copy_path.write_text(COPY, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", str(orig_path), str(copy_path), str(ans_path)],
    )

    main.main()

    output = capsys.readouterr().out
    assert "重复率：" in output
    assert "答案已写入文件" in output
    assert ans_path.read_text(encoding="utf-8") == "0.61"


def test_too_many_arguments(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "orig.txt", "copy.txt", "ans.txt", "extra.txt"],
    )

    main.main()

    assert "用法:" in capsys.readouterr().out


def test_read_text_gbk_open_oserror(capsys):
    first_error = UnicodeDecodeError(
        "utf-8",
        b"\xff",
        0,
        1,
        "invalid utf-8",
    )
    second_error = OSError("mock open error")

    with patch("builtins.open", side_effect=[first_error, second_error]):
        result = main.read_text("ignored.txt")

    assert result == ""
    assert "读取文件失败" in capsys.readouterr().out
