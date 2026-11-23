"""Provision a narrowly scoped Windows Firewall rule before starting LAN inference.

Only the LLM API port is opened to LocalSubnet on Domain/Private profiles.
Manager port 8766 is never opened. UAC is requested only when a rule change is
needed; refusing elevation fails closed (the batch does not start the LAN API).
"""
from __future__ import annotations

import base64
import ctypes
import os
import platform
import subprocess
import sys

import mini_runtime as rt

RULE_NAME = "Inkling Mini LLM API (LocalSubnet)"
POWERSHELL = "powershell.exe"


def encoded(script):
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def run_ps(script, elevated=False):
    payload = encoded(script)
    if elevated:
        # All data are generated from validated numeric port and fixed constants.
        outer = (
            "$p=Start-Process -FilePath 'powershell.exe' -Verb RunAs -Wait "
            "-PassThru -ArgumentList '-NoProfile -NonInteractive -EncodedCommand " +
            payload + "'; exit $p.ExitCode"
        )
        args = [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded(outer)]
    else:
        args = [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", payload]
    return subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=120)


def check_rule_script(port):
    return (
        "$ErrorActionPreference='Stop'; "
        f"$rule=Get-NetFirewallRule -DisplayName '{RULE_NAME}' -ErrorAction SilentlyContinue; "
        "if(-not $rule) { exit 3 }; "
        "$pf=$rule | Get-NetFirewallPortFilter; "
        "$af=$rule | Get-NetFirewallAddressFilter; "
        f"if($rule.Enabled -eq 'True' -and $rule.Direction -eq 'Inbound' -and "
        "$rule.Action -eq 'Allow' -and "
        "($rule.Profile -band [Microsoft.PowerShell.Cmdletization.GeneratedTypes.NetSecurity.Profile]::Domain) -and "
        "($rule.Profile -band [Microsoft.PowerShell.Cmdletization.GeneratedTypes.NetSecurity.Profile]::Private) -and "
        "$pf.Protocol -eq 'TCP' -and "
        f"$pf.LocalPort -eq '{port}' -and "
        "$af.RemoteAddress -eq 'LocalSubnet') {exit 0} else {exit 3}"
    )


def install_rule_script(port):
    return (
        "$ErrorActionPreference='Stop'; "
        f"$name='{RULE_NAME}'; "
        "Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue | "
        "Remove-NetFirewallRule -ErrorAction Stop; "
        f"New-NetFirewallRule -DisplayName $name -Direction Inbound "
        f"-Protocol TCP -LocalPort {port} -RemoteAddress LocalSubnet "
        "-Profile Domain,Private -Action Allow -Enabled True | Out-Null; "
        "exit 0"
    )


def ensure_firewall(port, executor=run_ps):
    if type(port) is not int or not 1024 <= port <= 65535 or port == 8766:
        raise ValueError("Invalid API port")
    check = executor(check_rule_script(port), elevated=False)
    if check.returncode == 0:
        return "existing"
    try:
        admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except AttributeError:
        admin = False
    result = executor(install_rule_script(port), elevated=not admin)
    if result.returncode != 0:
        raise RuntimeError("Windows UAC denied or firewall rule creation failed. " +
                           (result.stderr.strip() or result.stdout.strip())[:500])
    verify = executor(check_rule_script(port), elevated=False)
    if verify.returncode != 0:
        raise RuntimeError("Firewall rule was not verified; refusing LAN startup.")
    return "created"


def main():
    if platform.system() != "Windows":
        print("Windows firewall provisioning requires Windows Server x64.", file=sys.stderr)
        return 1
    cfg = rt.load_config()
    if not cfg["lan"]:
        print("LAN mode disabled: no inbound firewall rule needed.")
        return 0
    try:
        result = ensure_firewall(cfg["port"])
        print(f"LAN firewall {result}: TCP {cfg['port']}, Domain/Private, LocalSubnet only.")
        return 0
    except Exception as exc:
        print(f"LAN firewall setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
