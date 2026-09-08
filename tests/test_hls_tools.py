'''
Test HLS tool discovery and build command construction
'''

import stat
import pytest
import conifer.backends.common as common
from conifer.backends.common import get_hls, get_hls_build_command, get_xilinx_version


def make_stub(bindir, name, output=''):
    '''Create a fake tool executable to be discovered on PATH, echoing output on any invocation'''
    path = bindir / name
    path.write_text(f'#!/bin/sh\necho "{output}"\nexit 0\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


# https://docs.pytest.org/en/stable/reference/reference.html#pytest.MonkeyPatch.setenv
@pytest.fixture
def bindir(tmp_path, monkeypatch):
    d = tmp_path / 'bin'
    d.mkdir()
    monkeypatch.setenv('PATH', str(d))
    common._version_cache.clear()  # get_xilinx_version() is memoised per tool_exe across tests
    return d


def test_no_tool_in_path(bindir):
    assert get_hls() is None


def test_discover_vitis_unified(bindir):
    # Vitis 2025.1 and later ship only the unified CLI
    make_stub(bindir, 'vitis-run')
    assert get_hls() == 'vitis-run'


def test_prefer_classic_vitis_hls(bindir):
    # both tools are available from Vitis 2023.2 to 2024.2, prefer the classic one
    make_stub(bindir, 'vitis-run')
    make_stub(bindir, 'vitis_hls')
    assert get_hls() == 'vitis_hls'


def test_build_commands():
    assert get_hls_build_command('vivado_hls', 'build_hls.tcl') == 'vivado_hls -f build_hls.tcl'
    assert get_hls_build_command('vitis_hls', 'build_hls.tcl') == 'vitis_hls -f build_hls.tcl'
    assert get_hls_build_command('vitis-run', 'build_hls.tcl') == 'vitis-run --mode hls --tcl build_hls.tcl'


def test_get_xilinx_version_no_tool(bindir):
    assert get_xilinx_version() is None


def test_get_xilinx_version(bindir):
    make_stub(bindir, 'vitis_hls', 'Vitis(TM) HLS\nvitis_hls v2024.1 (64-bit)\nCopyright 1986-2024 ...')
    assert get_xilinx_version() == '2024.1'


def test_get_xilinx_version_unparseable(bindir):
    make_stub(bindir, 'vitis_hls', 'no version info here')
    assert get_xilinx_version() is None
