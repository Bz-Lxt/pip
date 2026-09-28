"""Tests for all things related to the configuration"""

import os
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pip._internal.configuration import (
    Configuration,
    _Edit,
    _ExclusiveFileLock,
    _patch_config_text,
    get_configuration_files,
    kinds,
)
from pip._internal.exceptions import ConfigurationError
from pip._internal.utils.compat import WINDOWS

from tests.lib.configuration_helpers import ConfigurationMixin


class TestConfigurationLoading(ConfigurationMixin):
    def test_global_loading(self) -> None:
        self.patch_configuration(kinds.GLOBAL, {"test.hello": "1"})

        self.configuration.load()
        assert self.configuration.get_value("test.hello") == "1"

    def test_user_loading(self) -> None:
        self.patch_configuration(kinds.USER, {"test.hello": "2"})

        self.configuration.load()
        assert self.configuration.get_value("test.hello") == "2"

    def test_site_loading(self) -> None:
        self.patch_configuration(kinds.SITE, {"test.hello": "3"})

        self.configuration.load()
        assert self.configuration.get_value("test.hello") == "3"

    def test_environment_config_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        contents = """
            [test]
            hello = 4
        """

        with self.tmpfile(contents) as config_file:
            monkeypatch.setenv("PIP_CONFIG_FILE", config_file)

            self.configuration.load()
            assert (
                self.configuration.get_value("test.hello") == "4"
            ), self.configuration._config

    def test_environment_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PIP_HELLO", "5")

        self.configuration.load()
        assert self.configuration.get_value(":env:.hello") == "5"

    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_environment_var_does_not_load_lowercase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("pip_hello", "5")

        self.configuration.load()
        with pytest.raises(ConfigurationError):
            self.configuration.get_value(":env:.hello")

    def test_environment_var_does_not_load_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PIP_VERSION", "True")

        self.configuration.load()

        with pytest.raises(ConfigurationError):
            self.configuration.get_value(":env:.version")

    def test_environment_config_errors_if_malformed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contents = """
            test]
            hello = 4
        """
        with self.tmpfile(contents) as config_file:
            monkeypatch.setenv("PIP_CONFIG_FILE", config_file)
            with pytest.raises(ConfigurationError) as err:
                self.configuration.load()

        assert "section header" in str(err.value)  # error kind
        assert "1" in str(err.value)  # line number
        assert config_file in str(err.value) or repr(config_file) in str(  # file name
            err.value
        )

    def test_no_such_key_error_message_no_command(self) -> None:
        self.configuration.load_only = kinds.GLOBAL
        self.configuration.load()
        expected_msg = (
            "Key does not contain dot separated section and key. "
            "Perhaps you wanted to use 'global.index-url' instead?"
        )
        pat = f"^{re.escape(expected_msg)}$"
        with pytest.raises(ConfigurationError, match=pat):
            self.configuration.get_value("index-url")

    def test_no_such_key_error_message_missing_option(self) -> None:
        self.configuration.load_only = kinds.GLOBAL
        self.configuration.load()
        expected_msg = "No such key - global.index-url"
        pat = f"^{re.escape(expected_msg)}$"
        with pytest.raises(ConfigurationError, match=pat):
            self.configuration.get_value("global.index-url")


class TestConfigurationPrecedence(ConfigurationMixin):
    # Tests for methods to that determine the order of precedence of
    # configuration options

    def test_env_overrides_site(self) -> None:
        self.patch_configuration(kinds.SITE, {"test.hello": "1"})
        self.patch_configuration(kinds.ENV, {"test.hello": "0"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "0"

    def test_env_overrides_user(self) -> None:
        self.patch_configuration(kinds.USER, {"test.hello": "2"})
        self.patch_configuration(kinds.ENV, {"test.hello": "0"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "0"

    def test_env_overrides_global(self) -> None:
        self.patch_configuration(kinds.GLOBAL, {"test.hello": "3"})
        self.patch_configuration(kinds.ENV, {"test.hello": "0"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "0"

    def test_site_overrides_user(self) -> None:
        self.patch_configuration(kinds.USER, {"test.hello": "2"})
        self.patch_configuration(kinds.SITE, {"test.hello": "1"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "1"

    def test_site_overrides_global(self) -> None:
        self.patch_configuration(kinds.GLOBAL, {"test.hello": "3"})
        self.patch_configuration(kinds.SITE, {"test.hello": "1"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "1"

    def test_user_overrides_global(self) -> None:
        self.patch_configuration(kinds.GLOBAL, {"test.hello": "3"})
        self.patch_configuration(kinds.USER, {"test.hello": "2"})
        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "2"

    def test_env_not_overridden_by_environment_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_configuration(kinds.ENV, {"test.hello": "1"})
        monkeypatch.setenv("PIP_HELLO", "5")

        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "1"
        assert self.configuration.get_value(":env:.hello") == "5"

    def test_site_not_overridden_by_environment_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_configuration(kinds.SITE, {"test.hello": "2"})
        monkeypatch.setenv("PIP_HELLO", "5")

        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "2"
        assert self.configuration.get_value(":env:.hello") == "5"

    def test_user_not_overridden_by_environment_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_configuration(kinds.USER, {"test.hello": "3"})
        monkeypatch.setenv("PIP_HELLO", "5")

        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "3"
        assert self.configuration.get_value(":env:.hello") == "5"

    def test_global_not_overridden_by_environment_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.patch_configuration(kinds.GLOBAL, {"test.hello": "4"})
        monkeypatch.setenv("PIP_HELLO", "5")

        self.configuration.load()

        assert self.configuration.get_value("test.hello") == "4"
        assert self.configuration.get_value(":env:.hello") == "5"


class TestConfigurationModification(ConfigurationMixin):
    # Tests for methods to that modify the state of a Configuration

    def test_no_specific_given_modification(self) -> None:
        self.configuration.load()

        with pytest.raises(ConfigurationError):
            self.configuration.set_value("test.hello", "10")

    def test_site_modification(self) -> None:
        self.configuration.load_only = kinds.SITE
        self.configuration.load()

        # Mock out the method
        mymock = MagicMock(spec=self.configuration._mark_as_modified)
        # https://github.com/python/mypy/issues/2427
        self.configuration._mark_as_modified = mymock  # type: ignore[method-assign]

        self.configuration.set_value("test.hello", "10")

        # get the path to site config file
        assert mymock.call_count == 1
        assert mymock.call_args[0][0] == (get_configuration_files()[kinds.SITE][0])

    def test_user_modification(self) -> None:
        # get the path to local config file
        self.configuration.load_only = kinds.USER
        self.configuration.load()

        # Mock out the method
        mymock = MagicMock(spec=self.configuration._mark_as_modified)
        # https://github.com/python/mypy/issues/2427
        self.configuration._mark_as_modified = mymock  # type: ignore[method-assign]

        self.configuration.set_value("test.hello", "10")

        # get the path to user config file
        assert mymock.call_count == 1
        assert mymock.call_args[0][0] == (
            # Use new config file
            get_configuration_files()[kinds.USER][1]
        )

    def test_global_modification(self) -> None:
        # get the path to local config file
        self.configuration.load_only = kinds.GLOBAL
        self.configuration.load()

        # Mock out the method
        mymock = MagicMock(spec=self.configuration._mark_as_modified)
        # https://github.com/python/mypy/issues/2427
        self.configuration._mark_as_modified = mymock  # type: ignore[method-assign]

        self.configuration.set_value("test.hello", "10")

        # get the path to user config file
        assert mymock.call_count == 1
        assert mymock.call_args[0][0] == (get_configuration_files()[kinds.GLOBAL][-1])

    def test_normalization(self) -> None:
        # underscores and dashes can be used interchangeably.
        # internally, underscores get converted into dashes before reading/writing file
        self.configuration.load_only = kinds.GLOBAL
        self.configuration.load()
        self.configuration.set_value("global.index_url", "example.org")
        assert self.configuration.get_value("global.index_url") == "example.org"
        assert self.configuration.get_value("global.index-url") == "example.org"
        self.configuration.unset_value("global.index-url")
        pat = r"^No such key - global\.index-url$"
        with pytest.raises(ConfigurationError, match=pat):
            self.configuration.get_value("global.index-url")


class TestInlineConfigEditing:
    """Surgical, byte-preserving edits performed via Configuration.save()."""

    original = textwrap.dedent("""\
        # top comment, must be preserved
        [global]
        # important option!
        timeout = 60
        index-url = https://example.org/simple/

        [freeze]
        ; semi-colon comment
        timeout=10
        """)

    def _configuration(self, config_file: Path) -> Configuration:
        config = Configuration(isolated=False, load_only=kinds.USER)
        parser = config._load_file(kinds.USER, str(config_file))
        config._parsers[kinds.USER].append((str(config_file), parser))
        return config

    def test_set_rewrites_only_the_target_line(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")
        config.save()

        expected = self.original.replace("timeout = 60", "timeout = 90")
        assert config_file.read_text() == expected

    def test_set_preserves_compact_separator_style(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("freeze.timeout", "20")
        config.save()

        # The "key=value" style of the freeze section is kept.
        assert config_file.read_text() == self.original.replace(
            "timeout=10", "timeout=20"
        )

    def test_set_inserts_option_into_existing_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.new-key", "value")
        config.save()

        expected = textwrap.dedent("""\
            # top comment, must be preserved
            [global]
            # important option!
            timeout = 60
            index-url = https://example.org/simple/
            new-key = value

            [freeze]
            ; semi-colon comment
            timeout=10
            """)
        assert config_file.read_text() == expected

    def test_set_appends_a_new_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("download.timeout", "5")
        config.save()

        expected = self.original + textwrap.dedent("""\

            [download]
            timeout = 5
            """)
        assert config_file.read_text() == expected

    def test_set_creates_missing_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"

        config = self._configuration(config_file)
        config.set_value("global.index-url", "https://example.org/")
        config.save()

        assert config_file.read_text() == (
            "[global]\nindex-url = https://example.org/\n"
        )

    def test_multiple_edits_in_one_session(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.first", "1")
        config.set_value("global.second", "2")
        config.save()

        content = config_file.read_text()
        assert "first = 1\n" in content
        assert "second = 2\n" in content
        assert content.count("[global]") == 1
        # Untouched bytes are still intact.
        assert "# top comment, must be preserved\n" in content

    def test_unset_removes_option_and_empty_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.unset_value("freeze.timeout")
        config.save()

        expected = textwrap.dedent("""\
            # top comment, must be preserved
            [global]
            # important option!
            timeout = 60
            index-url = https://example.org/simple/

            ; semi-colon comment
            """)
        assert config_file.read_text() == expected

    def test_unset_drops_multiline_continuation(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text("[global]\nkey = old\n line-one\n line-two\nother = 1\n")

        config = self._configuration(config_file)
        config.unset_value("global.key")
        config.save()

        assert config_file.read_text() == "[global]\nother = 1\n"

    def test_set_replacing_multiline_value_drops_continuation(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text("[global]\nkey = old\n line\nother = 1\n")

        config = self._configuration(config_file)
        config.set_value("global.key", "new")
        config.save()

        assert config_file.read_text() == "[global]\nkey = new\nother = 1\n"

    def test_preserves_crlf_line_endings(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_bytes(b"[global]\r\ntimeout = 60\r\n")

        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")
        config.save()

        assert config_file.read_bytes() == b"[global]\r\ntimeout = 90\r\n"

    def test_idempotent_set_does_not_touch_the_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.timeout", "60")

        replacements: list[str] = []

        def fake_atomic_replace(*args: object, **kwargs: object) -> None:
            replacements.append("x")

        config._atomic_replace = fake_atomic_replace  # type: ignore[method-assign]
        config.save()

        assert replacements == []
        assert config_file.read_bytes() == self.original.encode()

    def test_repeated_set_across_instances_is_stable(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")
        config.save()
        after_first = config_file.read_bytes()

        # A second, identical command (e.g. a re-run by the user).
        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")
        config.save()

        assert config_file.read_bytes() == after_first

    def test_conflict_is_detected_and_file_is_preserved(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")

        # Another process modifies the file before we save.
        external_content = self.original.replace("timeout = 60", "timeout = 61")
        config_file.write_text(external_content)

        with pytest.raises(ConfigurationError, match="modified by another process"):
            config.save()

        # No truncation and no overwrite of the external change.
        assert config_file.read_text() == external_content

    def test_conflict_when_file_is_created_concurrently(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"

        config = self._configuration(config_file)  # File did not exist.
        config.set_value("global.timeout", "90")

        config_file.write_text("[global]\ntimeout = 1\n")

        with pytest.raises(ConfigurationError, match="modified by another process"):
            config.save()

        assert config_file.read_text() == "[global]\ntimeout = 1\n"

    def test_write_failure_leaves_original_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "pip.conf"
        config_file.write_text(self.original)

        config = self._configuration(config_file)
        config.set_value("global.timeout", "90")

        def failing_replace(src: str, dst: str) -> None:
            raise OSError("simulated failure")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(ConfigurationError):
            config.save()

        # Original bytes survive and no temp file is left behind.
        assert config_file.read_text() == self.original
        assert list(tmp_path.glob(".pip.conf.*.tmp")) == []


class TestInlineConfigEditor:
    """Direct tests for the text-level editor."""

    def test_value_with_delimiters_and_comment_chars(self) -> None:
        text = "[global]\nkey = old : value = x # not a comment\n"
        result = _patch_config_text(
            text, [_Edit("set", "global", "key", "new : v = y # keep")]
        )
        assert result == "[global]\nkey = new : v = y # keep\n"

    def test_colon_separator_preserved(self) -> None:
        text = "[global]\nkey: old\n"
        result = _patch_config_text(text, [_Edit("set", "global", "key", "new")])
        assert result == "[global]\nkey: new\n"

    def test_continuation_line_is_not_matched_as_option(self) -> None:
        text = "[global]\nfirst = 1\n key = 2\n"
        result = _patch_config_text(text, [_Edit("set", "global", "key", "x")])
        # "key" only exists as a continuation; a real option is inserted.
        assert result == "[global]\nfirst = 1\n key = 2\nkey = x\n"

    def test_unset_keeps_surrounding_comments(self) -> None:
        text = "# pre\n[global]\n# inner\nkey = v\n# post\n"
        result = _patch_config_text(text, [_Edit("unset", "global", "key")])
        assert result == "# pre\n# inner\n# post\n"


@pytest.mark.skipif(WINDOWS, reason="fcntl.flock is POSIX-only")
class TestExclusiveFileLock:
    def test_lock_excludes_other_open_descriptions(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"

        with _ExclusiveFileLock(str(config_file)):
            import fcntl

            other_fd = os.open(str(config_file) + ".lock", os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(other_fd)

    def test_lock_is_released_on_exit(self, tmp_path: Path) -> None:
        config_file = tmp_path / "pip.conf"

        with _ExclusiveFileLock(str(config_file)):
            pass

        import fcntl

        other_fd = os.open(str(config_file) + ".lock", os.O_RDWR)
        try:
            fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other_fd)
