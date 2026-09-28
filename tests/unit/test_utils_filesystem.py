import errno
import os
import threading
from pathlib import Path

import pytest

from pip._internal.utils import filesystem
from pip._internal.utils.filesystem import (
    PublishedFile,
    _subdirs_without_generic,
    files_are_identical,
    publish_data,
    publish_file,
    subdirs_without_files,
    subdirs_without_wheels,
)


class TestFilesAreIdentical:
    def test_identical_contents(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"same contents")
        b.write_bytes(b"same contents")
        assert files_are_identical(str(a), str(b))

    def test_different_sizes(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"short")
        b.write_bytes(b"a longer payload")
        assert not files_are_identical(str(a), str(b))

    def test_same_size_different_contents(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"abcd")
        b.write_bytes(b"abce")
        assert not files_are_identical(str(a), str(b))

    def test_missing_file_is_not_identical(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.write_bytes(b"data")
        assert not files_are_identical(str(a), str(tmp_path / "missing"))


def _hash_alternative(dest: str, src: str, index: int) -> str:
    """Deterministic alternative name used by the publish tests."""
    stem, ext = os.path.splitext(dest)
    suffix = "" if index == 0 else f"-{index}"
    return f"{stem}.alt{suffix}{ext}"


class TestPublishFile:
    def test_creates_destination_when_absent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"payload")
        result = publish_file(str(src), str(dest), _hash_alternative)
        assert result.path == str(dest)
        assert result.created is True
        assert dest.read_bytes() == b"payload"

    def test_reuses_identical_destination(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"payload")
        dest.write_bytes(b"payload")
        inode_before = dest.stat().st_ino
        result = publish_file(str(src), str(dest), _hash_alternative)
        assert result.path == str(dest)
        assert result.created is False
        assert dest.stat().st_ino == inode_before

    def test_conflicting_contents_are_protected(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"new contents")
        dest.write_bytes(b"original contents")

        result = publish_file(str(src), str(dest), _hash_alternative)

        assert result.created is True
        assert result.path != str(dest)
        # The original file is left exactly where it was.
        assert dest.read_bytes() == b"original contents"
        # The new contents land at the deterministic alternative path.
        assert Path(result.path).read_bytes() == b"new contents"
        assert result.path == str(tmp_path / "dest.alt")

    def test_conflict_without_alternative_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"new contents")
        dest.write_bytes(b"original contents")
        with pytest.raises(FileExistsError):
            publish_file(str(src), str(dest))

    def test_retry_is_idempotent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"new contents")
        dest.write_bytes(b"original contents")

        first = publish_file(str(src), str(dest), _hash_alternative)
        second = publish_file(str(src), str(dest), _hash_alternative)

        assert first.path == second.path
        assert second.created is False
        names = {
            p.name for p in tmp_path.iterdir() if p.name not in ("home", "fake-root")
        }
        assert names == {"dest", "dest.alt", "src"}

    def test_distinct_contents_get_distinct_alternatives(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"
        dest.write_bytes(b"original")
        sources = []
        for i, payload in enumerate([b"one", b"two"]):
            src = tmp_path / f"src{i}"
            src.write_bytes(payload)
            sources.append(src)

        first = publish_file(str(sources[0]), str(dest), _hash_alternative)
        second = publish_file(str(sources[1]), str(dest), _hash_alternative)

        assert first.path != second.path
        assert dest.read_bytes() == b"original"
        assert Path(first.path).read_bytes() == b"one"
        assert Path(second.path).read_bytes() == b"two"

    def test_cross_device_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"payload")
        real_link = os.link

        def link_across_devices(source: str, target: str) -> None:
            # Only the direct link from src crosses devices; the staged
            # adjacent copy must link successfully.
            if os.path.abspath(source) == str(src):
                raise OSError(errno.EXDEV, "cross-device link")
            real_link(source, target)

        monkeypatch.setattr(filesystem.os, "link", link_across_devices)

        result = publish_file(str(src), str(dest), _hash_alternative)
        assert result.path == str(dest)
        assert dest.read_bytes() == b"payload"
        # No staging temp files are left behind (the test harness may add
        # unrelated "home"/"fake-root" isolation directories to tmp_path).
        names = {
            p.name for p in tmp_path.iterdir() if p.name not in ("home", "fake-root")
        }
        assert names == {"src", "dest"}

    def test_concurrent_identical_publishes_converge(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.write_bytes(b"same payload for everyone")
        dest = tmp_path / "dest"
        results: list[str] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def publish() -> None:
            try:
                result = publish_file(str(src), str(dest), _hash_alternative)
                with lock:
                    results.append(result.path)
            except BaseException as exc:  # pragma: no cover - reported below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=publish) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(results) == 16
        assert set(results) == {str(dest)}

    def test_concurrent_distinct_publishes_are_stable(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"
        dest.write_bytes(b"original")
        sources: dict[int, str] = {}
        for i in range(8):
            src = tmp_path / f"src{i}"
            src.write_bytes(f"payload-{i}".encode())
            sources[i] = str(src)
        results: list[str] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def publish(index: int) -> None:
            try:
                result = publish_file(sources[index], str(dest), _hash_alternative)
                with lock:
                    results.append(result.path)
            except BaseException as exc:  # pragma: no cover - reported below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=publish, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(results) == len(set(results)) == 8
        assert dest.read_bytes() == b"original"
        # Re-running the same publishes must reuse every existing file.
        rerun = [
            publish_file(sources[i], str(dest), _hash_alternative) for i in range(8)
        ]
        assert all(not result.created for result in rerun)
        assert sorted(result.path for result in rerun) == sorted(results)


class TestPublishFileIncomingWins:
    def test_replaces_conflicting_file_and_preserves_occupant(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"verified new contents")
        dest.write_bytes(b"old contents")

        result = publish_file(
            str(src), str(dest), _hash_alternative, incoming_wins=True
        )

        assert result.path == str(dest)
        assert result.created is True
        assert result.displaced_to == str(tmp_path / "dest.alt")
        # The authoritative incoming contents take the canonical name...
        assert dest.read_bytes() == b"verified new contents"
        # ...and the displaced contents survive at the backup path.
        assert Path(result.displaced_to).read_bytes() == b"old contents"
        # The source is consumed by neither move nor deletion.
        assert src.read_bytes() == b"verified new contents"

    def test_identical_contents_are_reused(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"same")
        dest.write_bytes(b"same")

        result = publish_file(
            str(src), str(dest), _hash_alternative, incoming_wins=True
        )

        assert result.created is False
        assert result.displaced_to is None
        assert not (tmp_path / "dest.alt").exists()

    def test_retry_is_idempotent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_bytes(b"new")
        dest.write_bytes(b"old")

        first = publish_file(str(src), str(dest), _hash_alternative, incoming_wins=True)
        second = publish_file(
            str(src), str(dest), _hash_alternative, incoming_wins=True
        )

        assert first.path == second.path == str(dest)
        assert first.displaced_to == str(tmp_path / "dest.alt")
        assert second.created is False
        assert second.displaced_to is None
        backup = tmp_path / "dest.alt"
        assert backup.read_bytes() == b"old"
        assert dest.read_bytes() == b"new"

    def test_successive_conflicts_preserve_each_occupant(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"
        dest.write_bytes(b"v0")
        sources = []
        for i, payload in enumerate([b"v1", b"v2"]):
            src = tmp_path / f"src{i}"
            src.write_bytes(payload)
            sources.append(src)

        publish_file(str(sources[0]), str(dest), _hash_alternative, incoming_wins=True)
        second = publish_file(
            str(sources[1]), str(dest), _hash_alternative, incoming_wins=True
        )

        assert dest.read_bytes() == b"v2"
        assert (tmp_path / "dest.alt").read_bytes() == b"v0"
        assert second.displaced_to == str(tmp_path / "dest.alt-1")
        assert Path(second.displaced_to).read_bytes() == b"v1"

    def test_concurrent_identical_publishes_converge(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.write_bytes(b"verified new contents")
        dest = tmp_path / "dest"
        dest.write_bytes(b"old contents")
        results: list[PublishedFile] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def publish() -> None:
            try:
                result = publish_file(
                    str(src), str(dest), _hash_alternative, incoming_wins=True
                )
                with lock:
                    results.append(result)
            except BaseException as exc:  # pragma: no cover - reported below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=publish) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(results) == 16
        assert {result.path for result in results} == {str(dest)}
        assert dest.read_bytes() == b"verified new contents"
        # The displaced occupant is backed up exactly once.
        backups = [
            p for p in tmp_path.glob("dest.alt*") if p.name not in ("home", "fake-root")
        ]
        assert len(backups) == 1
        assert backups[0].read_bytes() == b"old contents"
        # A second wave reuses the converged layout.
        rerun = [
            publish_file(str(src), str(dest), _hash_alternative, incoming_wins=True)
            for _ in range(8)
        ]
        assert all(not result.created for result in rerun)


class TestPublishData:
    def test_creates_and_reuses(self, tmp_path: Path) -> None:
        dest = tmp_path / "file"
        first = publish_data(str(dest), b"payload")
        second = publish_data(str(dest), b"payload")
        assert first.created is True
        assert second.created is False
        assert second.path == str(dest)
        assert dest.read_bytes() == b"payload"

    def test_conflict_uses_alternative(self, tmp_path: Path) -> None:
        dest = tmp_path / "file"
        dest.write_bytes(b"original")
        result = publish_data(str(dest), b"new", _hash_alternative)
        assert result.path == str(tmp_path / "file.alt")
        assert dest.read_bytes() == b"original"
        assert Path(result.path).read_bytes() == b"new"


def make_file(path: str) -> None:
    Path(path).touch()


def make_valid_symlink(path: str) -> None:
    target = path + "1"
    make_file(target)
    os.symlink(target, path)


def make_broken_symlink(path: str) -> None:
    os.symlink("foo", path)


def make_dir(path: str) -> None:
    os.mkdir(path)


class TestSubdirsWithoutGeneric:
    """Tests for _subdirs_without_generic."""

    def test_yieled_in_reverse_order(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "empty"
        deepest_dir = test_dir / "a" / "b" / "c" / "d"
        deepest_dir.mkdir(parents=True)

        result = list(
            _subdirs_without_generic(str(test_dir), lambda root, filenames: False)
        )
        assert [
            deepest_dir,
            deepest_dir.parent,
            deepest_dir.parent.parent,
            deepest_dir.parent.parent.parent,
            test_dir,
        ] == result


class TestSubdirsWithoutFiles:
    """Tests for subdirs_without_files function."""

    def test_empty_directory_tree(self, tmp_path: Path) -> None:
        """An empty directory should yield itself."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        result = list(subdirs_without_files(str(test_dir)))

        assert result == [test_dir]

    def test_directory_with_file(self, tmp_path: Path) -> None:
        """A directory with a file should not be yielded."""
        test_dir = tmp_path / "with_file"
        test_dir.mkdir()
        (test_dir / "file.txt").touch()

        result = list(subdirs_without_files(str(test_dir)))

        assert result == []

    def test_nested_empty_directories(self, tmp_path: Path) -> None:
        """Nested empty directories should all be yielded."""
        test_dir = tmp_path / "root"
        (test_dir / "a" / "b" / "c").mkdir(parents=True)
        (test_dir / "d" / "e").mkdir(parents=True)

        result = list(subdirs_without_files(str(test_dir)))

        # Should yield all directories since none have files
        assert len(result) == 6  # root, a, a/b, a/b/c, d, d/e
        assert test_dir in result

    def test_mixed_empty_and_non_empty(self, tmp_path: Path) -> None:
        """Only empty subdirectories should be yielded."""
        test_dir = tmp_path / "mixed"
        empty1 = test_dir / "empty1"
        empty2 = test_dir / "empty2"
        with_file = test_dir / "with_file"

        empty1.mkdir(parents=True)
        empty2.mkdir(parents=True)
        with_file.mkdir(parents=True)
        (with_file / "file.txt").touch()

        result = list(subdirs_without_files(str(test_dir)))

        assert empty1 in result
        assert empty2 in result
        assert with_file not in result
        assert test_dir not in result  # root has subdirs with files

    def test_deeply_nested_file_marks_parents(self, tmp_path: Path) -> None:
        """A file deep in the tree marks all parent directories as non-empty."""
        test_dir = tmp_path / "deep"
        deep_dir = test_dir / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        (deep_dir / "file.txt").touch()

        # Create another branch that's empty
        empty_branch = test_dir / "empty"
        empty_branch.mkdir()

        result = list(subdirs_without_files(str(test_dir)))

        # Only the empty branch should be yielded
        assert result == [empty_branch]

    def test_multiple_files_at_various_levels(self, tmp_path: Path) -> None:
        """Files at various levels should be handled correctly."""
        test_dir = tmp_path / "various"
        (test_dir / "a" / "b").mkdir(parents=True)
        (test_dir / "c" / "d").mkdir(parents=True)
        (test_dir / "e").mkdir(parents=True)

        # File in a/b
        (test_dir / "a" / "b" / "file1.txt").touch()
        # File in c (not in c/d)
        (test_dir / "c" / "file2.txt").touch()
        # e is empty

        result = list(subdirs_without_files(str(test_dir)))

        # Only c/d and e should be yielded (c/d has no files, e is empty)
        assert test_dir / "c" / "d" in result
        assert test_dir / "e" in result
        assert len(result) == 2


class TestSubdirsWithoutWheels:
    """Tests for subdirs_without_wheels function."""

    def test_directory_with_wheel_file(self, tmp_path: Path) -> None:
        """A directory with a .whl file should not be yielded."""
        test_dir = tmp_path / "with_wheel"
        test_dir.mkdir()
        (test_dir / "package-1.0-py3-none-any.whl").touch()

        result = list(subdirs_without_wheels(str(test_dir)))

        assert result == []

    def test_directory_with_non_wheel_files(self, tmp_path: Path) -> None:
        """A directory with non-wheel files should be yielded."""
        test_dir = tmp_path / "no_wheels"
        test_dir.mkdir()
        (test_dir / "metadata.json").touch()
        (test_dir / "readme.txt").touch()

        result = list(subdirs_without_wheels(str(test_dir)))

        assert result == [test_dir]

    def test_mixed_wheel_and_non_wheel_dirs(self, tmp_path: Path) -> None:
        """Only directories without wheels should be yielded."""
        test_dir = tmp_path / "mixed_wheels"
        has_wheel = test_dir / "has_wheel"
        no_wheel = test_dir / "no_wheel"

        has_wheel.mkdir(parents=True)
        no_wheel.mkdir(parents=True)

        (has_wheel / "package-1.0-py3-none-any.whl").touch()
        (no_wheel / "metadata.json").touch()

        result = list(subdirs_without_wheels(str(test_dir)))

        assert no_wheel in result
        assert has_wheel not in result
        assert test_dir not in result  # root has subdir with wheel

    def test_nested_wheel_marks_parents(self, tmp_path: Path) -> None:
        """A wheel file deep in the tree marks all parent directories."""
        test_dir = tmp_path / "nested_wheel"
        deep_dir = test_dir / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        (deep_dir / "package-1.0-py3-none-any.whl").touch()

        empty_branch = test_dir / "empty"
        empty_branch.mkdir()

        result = list(subdirs_without_wheels(str(test_dir)))

        # Only the empty branch should be yielded
        assert result == [empty_branch]

    def test_wheel_file_extension_matching(self, tmp_path: Path) -> None:
        """Only files ending with .whl should be considered wheels."""
        test_dir = tmp_path / "extensions"
        (test_dir / "subdir1").mkdir(parents=True)
        (test_dir / "subdir2").mkdir(parents=True)

        # Not a wheel (missing extension)
        (test_dir / "subdir1" / "package-1.0-py3-none-any").touch()
        # Not a wheel (.whl in middle of name)
        (test_dir / "subdir1" / "package.whl.backup").touch()

        # This is a wheel
        (test_dir / "subdir2" / "real-1.0-py3-none-any.whl").touch()

        result = list(subdirs_without_wheels(str(test_dir)))

        assert test_dir / "subdir1" in result
        assert test_dir / "subdir2" not in result

    def test_empty_directory_tree_no_wheels(self, tmp_path: Path) -> None:
        """Empty directories should be yielded as having no wheels."""
        test_dir = tmp_path / "empty_tree"
        (test_dir / "a" / "b").mkdir(parents=True)
        (test_dir / "c").mkdir(parents=True)

        result = list(subdirs_without_wheels(str(test_dir)))

        # All directories should be yielded since none have wheels
        assert len(result) == 4  # test_dir, a, a/b, c
        assert test_dir in result
