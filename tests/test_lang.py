# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dataset language detection."""

import pytest

from aevyra_reflex.lang import detect_language


class TestDetectLanguage:
    """Tests for detect_language() — Unicode script heuristics."""

    def test_empty_list_returns_english(self):
        assert detect_language([]) == "English"

    def test_empty_strings_return_english(self):
        assert detect_language(["", "   ", "\n"]) == "English"

    def test_ascii_english_text(self):
        samples = [
            "How do I configure retry logic for failed pipeline steps?",
            "What serialization format should I use for a production pipeline?",
            "How do I deduplicate records by a specific field?",
        ]
        assert detect_language(samples) == "English"

    def test_chinese_simplified(self):
        samples = [
            "如何配置失败管道步骤的重试逻辑？",
            "生产环境中应该使用什么序列化格式？",
            "如何根据特定字段对记录进行去重？",
        ]
        assert detect_language(samples) == "Chinese"

    def test_chinese_traditional(self):
        samples = [
            "如何配置失敗管道步驟的重試邏輯？",
            "生產環境中應該使用什麼序列化格式？",
        ]
        assert detect_language(samples) == "Chinese"

    def test_japanese(self):
        samples = [
            "パイプラインの再試行ロジックを設定するにはどうすればいいですか？",
            "本番環境ではどのシリアライゼーション形式を使うべきですか？",
        ]
        assert detect_language(samples) == "Japanese"

    def test_korean(self):
        samples = [
            "파이프라인 단계 실패 시 재시도 로직을 어떻게 구성하나요?",
            "프로덕션 파이프라인에는 어떤 직렬화 형식을 사용해야 하나요?",
        ]
        assert detect_language(samples) == "Korean"

    def test_arabic(self):
        samples = [
            "كيف يمكنني تكوين منطق إعادة المحاولة لخطوات الأنابيب الفاشلة؟",
            "ما تنسيق التسلسل الذي يجب استخدامه لأنبوب الإنتاج؟",
        ]
        assert detect_language(samples) == "Arabic"

    def test_russian(self):
        samples = [
            "Как настроить логику повторных попыток для неудачных шагов конвейера?",
            "Какой формат сериализации следует использовать для производственного конвейера?",
        ]
        assert detect_language(samples) == "Russian"

    def test_mixed_english_with_few_cjk_chars_stays_english(self):
        """A few foreign characters in English text must not trigger mislabelling."""
        samples = [
            'The class is called "数据流" in the docs.',
            "Call pipeline.tap() — see also: パイプライン",
            "Normal English question about the API.",
            "Another plain English sentence with no special characters.",
            "What does on_error='dead_letter' do?",
        ]
        result = detect_language(samples)
        # Below the 15% threshold → stays English
        assert result == "English"

    def test_predominantly_chinese_with_some_latin(self):
        """Mostly Chinese text with some ASCII punctuation → Chinese."""
        samples = [
            "如何配置API的retry参数？",   # mixed Chinese + ASCII "API" and "retry"
            "使用Pipeline(retry=RetryPolicy(...))来设置重试逻辑。",
        ]
        assert detect_language(samples) == "Chinese"

    def test_single_long_english_string(self):
        result = detect_language([
            "This is a very long English paragraph. " * 50
        ])
        assert result == "English"

    def test_returns_string(self):
        """detect_language always returns a non-empty string."""
        for texts in [[], ["hello"], ["こんにちは"], ["مرحبا"]]:
            result = detect_language(texts)
            assert isinstance(result, str)
            assert len(result) > 0
