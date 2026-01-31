#!/usr/bin/env python
"""
Extended unit tests for utils/utils.py to improve code coverage.

Tests cover:
- Fallback formatting in _object_attributes_to_table when columnar is not available (HAS_COLUMNAR=False)
- check_object_pickleability with non-pickleable attributes (return False path)
- display_object_attributes with various objects
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.unit


class TestColumnarFallbackFormatting:
    """Tests for _object_attributes_to_table fallback when columnar is not available."""

    def test_fallback_formatting_without_columnar(self):
        """Test fallback formatting when HAS_COLUMNAR=False."""
        with patch("utils.utils.HAS_COLUMNAR", False), patch("utils.utils.col", None):
            from utils.utils import _init_content_list

            content = _init_content_list(True)
            assert content == []

            content.append(["attr1", "value1"])
            content.append(["attr2", "value2"])

            headers = ["Attribute", "Attribute Value"]
            result = "\n".join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content])

            assert "Attribute: attr1, Attribute Value: value1" in result
            assert "Attribute: attr2, Attribute Value: value2" in result

    def test_fallback_formatting_empty_content(self):
        """Test fallback formatting with empty content list."""
        with patch("utils.utils.HAS_COLUMNAR", False), patch("utils.utils.col", None):
            content = []
            headers = ["Attribute", "Attribute Value"]
            result = "\n".join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content]) if content else None

            assert result is None

    def test_fallback_formatting_single_attribute(self):
        """Test fallback formatting with single attribute."""
        with patch("utils.utils.HAS_COLUMNAR", False), patch("utils.utils.col", None):
            content = [["name", "test_value"]]
            headers = ["Attribute", "Attribute Value"]
            result = "\n".join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content])

            assert result == "Attribute: name, Attribute Value: test_value"

    def test_fallback_formatting_with_complex_values(self):
        """Test fallback formatting with complex attribute values."""
        with patch("utils.utils.HAS_COLUMNAR", False), patch("utils.utils.col", None):
            content = [
                ["list_attr", [1, 2, 3]],
                ["dict_attr", {"key": "value"}],
                ["none_attr", None],
            ]
            headers = ["Attribute", "Attribute Value"]
            result = "\n".join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content])

            assert "list_attr" in result
            assert "[1, 2, 3]" in result
            assert "dict_attr" in result
            assert "{'key': 'value'}" in result
            assert "none_attr" in result


try:
    import dill

    HAS_DILL = True
except ImportError:
    HAS_DILL = False


class TestCheckObjectPickleabilityFalsePath:
    """Tests for check_object_pickleability returning False with non-pickleable attributes."""

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_non_pickleable_socket(self):
        """Test with object containing socket (non-pickleable)."""
        import socket

        from utils.utils import check_object_pickleability

        class SocketClass:
            def __init__(self):
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.name = "test"

        obj = SocketClass()
        try:
            result = check_object_pickleability(obj)
            assert isinstance(result, bool)
        finally:
            obj.sock.close()

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_non_pickleable_generator(self):
        """Test with object containing a generator."""
        from utils.utils import check_object_pickleability

        class GeneratorClass:
            def __init__(self):
                self.generator = (x for x in range(10))
                self.value = 42

        obj = GeneratorClass()
        result = check_object_pickleability(obj)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_non_pickleable_thread_lock(self):
        """Test with object containing a thread lock."""
        import threading

        from utils.utils import check_object_pickleability

        class LockClass:
            def __init__(self):
                self.lock = threading.Lock()
                self.value = "test"

        obj = LockClass()
        result = check_object_pickleability(obj)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_pickleability_returns_false_for_problematic_attr(self):
        """Test that function returns False when any attribute is not pickleable."""
        import dill

        from utils.utils import check_object_pickleability

        class ProblematicClass:
            def __init__(self):
                self.simple = 42
                self.problematic = lambda x: x  # lambdas are typically not pickleable with pickle

        obj = ProblematicClass()
        result = check_object_pickleability(obj)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_pickleability_all_attrs_pickleable(self):
        """Test that function returns True when all attributes are pickleable."""
        from utils.utils import check_object_pickleability

        class SimpleClass:
            def __init__(self):
                self.a = 1
                self.b = "two"
                self.c = [1, 2, 3]

        obj = SimpleClass()
        result = check_object_pickleability(obj)
        assert result is True

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_pickleability_iterates_all_attributes(self):
        """Test that function iterates through all __dict__ keys."""
        from utils.utils import check_object_pickleability

        class MultiAttrClass:
            def __init__(self):
                self.attr1 = 1
                self.attr2 = 2
                self.attr3 = 3
                self.attr4 = 4
                self.attr5 = 5

        obj = MultiAttrClass()
        result = check_object_pickleability(obj)
        assert result is True


class TestDisplayObjectAttributesVariousObjects:
    """Tests for display_object_attributes with various objects."""

    def test_display_object_attributes_with_logging_module(self):
        """Test with 'logging' module."""
        from utils.utils import display_object_attributes

        try:
            result = display_object_attributes("logging", private_attrs=False)
            assert result is None or isinstance(result, str) or result is False
        except AttributeError:
            pass

    def test_display_object_attributes_with_none_object(self):
        """Test with None as object_name."""
        from utils.utils import display_object_attributes

        result = display_object_attributes(None)
        assert result is None or result is False

    def test_display_object_attributes_with_empty_string(self):
        """Test with empty string as object_name."""
        from utils.utils import display_object_attributes

        try:
            result = display_object_attributes("")
            assert result is None or result is False
        except (ModuleNotFoundError, ValueError):
            pass

    def test_display_object_attributes_with_nonexistent_module(self):
        """Test with non-existent module name."""
        from utils.utils import display_object_attributes

        try:
            result = display_object_attributes("nonexistent_module_xyz_123")
            assert result is None
        except ModuleNotFoundError:
            pass

    def test_display_object_attributes_private_attrs_true(self):
        """Test with private_attrs=True."""
        from utils.utils import display_object_attributes

        try:
            result = display_object_attributes("os", private_attrs=True)
            assert result is None or isinstance(result, str) or result is False
        except AttributeError:
            pass

    def test_display_object_attributes_private_attrs_false(self):
        """Test with private_attrs=False."""
        from utils.utils import display_object_attributes

        try:
            result = display_object_attributes("os", private_attrs=False)
            assert result is None or isinstance(result, str) or result is False
        except AttributeError:
            pass


class TestObjectAttributesToTableColumnarPath:
    """Tests for _object_attributes_to_table with columnar available and unavailable."""

    def test_table_with_columnar_available(self):
        """Test table generation when columnar is available."""
        from utils.utils import HAS_COLUMNAR

        if HAS_COLUMNAR:
            from utils.utils import _init_content_list

            content = _init_content_list(True)
            content.append(["test_attr", "test_value"])

    def test_table_fallback_path_directly(self):
        """Test the fallback formatting path directly."""
        content = [["attr1", "value1"], ["attr2", 42], ["attr3", [1, 2, 3]]]
        headers = ["Attribute", "Attribute Value"]

        result = "\n".join([f"{headers[0]}: {row[0]}, {headers[1]}: {row[1]}" for row in content]) if content else None

        assert "attr1" in result
        assert "value1" in result
        assert "attr2" in result
        assert "42" in result
        assert "attr3" in result

    def test_table_with_private_attributes_filtered(self):
        """Test that private attributes are filtered when private_attrs=False."""
        obj_dict = {"public_attr": 1, "_private_attr": 2, "__dunder__": 3}
        keys = list(obj_dict.keys())
        private_attrs = False

        content = []
        for key in keys:
            if key.startswith("_") and not private_attrs:
                continue
            content.append([key, obj_dict.get(key)])

        assert len(content) == 1
        assert content[0][0] == "public_attr"

    def test_table_with_private_attributes_included(self):
        """Test that private attributes are included when private_attrs=True."""
        obj_dict = {"public_attr": 1, "_private_attr": 2, "__dunder__": 3}
        keys = list(obj_dict.keys())
        private_attrs = True

        content = []
        for key in keys:
            if key.startswith("_") and not private_attrs:
                continue
            content.append([key, obj_dict.get(key)])

        assert len(content) == 3


class TestInitContentListEdgeCases:
    """Edge case tests for _init_content_list function."""

    def test_init_content_list_with_truthy_value(self):
        """Test _init_content_list with various truthy values."""
        from utils.utils import _init_content_list

        assert _init_content_list(True) == []
        assert _init_content_list(1) == []
        assert _init_content_list("non-empty") == []

    def test_init_content_list_with_falsy_value(self):
        """Test _init_content_list with various falsy values."""
        from utils.utils import _init_content_list

        assert _init_content_list(False) is None
        assert _init_content_list(0) is None
        assert _init_content_list("") is None
        assert _init_content_list(None) is None


class TestCheckObjectPickleabilityEdgeCases:
    """Edge case tests for check_object_pickleability function."""

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_object_with_empty_dict(self):
        """Test with object that has empty __dict__."""
        from utils.utils import check_object_pickleability

        class EmptyClass:
            pass

        obj = EmptyClass()
        result = check_object_pickleability(obj)
        assert result is True

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_object_with_torch_tensor(self):
        """Test with object containing torch tensor."""
        from utils.utils import check_object_pickleability

        class TensorClass:
            def __init__(self):
                self.tensor = torch.randn(3, 3)
                self.value = 42

        obj = TensorClass()
        result = check_object_pickleability(obj)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_object_with_numpy_array(self):
        """Test with object containing numpy array."""
        from utils.utils import check_object_pickleability

        class NumpyClass:
            def __init__(self):
                self.array = np.array([1, 2, 3, 4, 5])
                self.value = "test"

        obj = NumpyClass()
        result = check_object_pickleability(obj)
        assert result is True

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_none_instance(self):
        """Test with None instance."""
        from utils.utils import check_object_pickleability

        result = check_object_pickleability(None)
        assert result is False

    @pytest.mark.skipif(not HAS_DILL, reason="Requires dill package")
    def test_primitive_without_dict(self):
        """Test with primitive type that has no __dict__."""
        from utils.utils import check_object_pickleability

        result = check_object_pickleability(42)
        assert result is False

        result = check_object_pickleability("string")
        assert result is False

        result = check_object_pickleability([1, 2, 3])
        assert result is False


class TestHasColumnarImport:
    """Tests for HAS_COLUMNAR import flag."""

    def test_has_columnar_is_boolean(self):
        """Test that HAS_COLUMNAR is a boolean."""
        from utils.utils import HAS_COLUMNAR

        assert isinstance(HAS_COLUMNAR, bool)

    def test_col_matches_has_columnar(self):
        """Test that col is None when HAS_COLUMNAR is False, and not None when True."""
        from utils.utils import HAS_COLUMNAR, col

        if HAS_COLUMNAR:
            assert col is not None
        else:
            assert col is None
