# Mutes every "Mini Motorways" audio session at the Windows level (Core Audio API),
# and prints each one's current output peak so the result is verifiable.
# Usage:  powershell -File mute_games.ps1 [-Unmute]
param([switch]$Unmute)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorCom {}

[ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int stateMask, out IntPtr devices);
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
}
[ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(ref Guid iid, int clsCtx, IntPtr activationParams, [MarshalAs(UnmanagedType.IUnknown)] out object iface);
}
[ComImport, Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionManager2 {
    int GetAudioSessionControl(IntPtr a, int b, out IntPtr c);
    int GetSimpleAudioVolume(IntPtr a, int b, out IntPtr c);
    int GetSessionEnumerator(out IAudioSessionEnumerator e);
}
[ComImport, Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionEnumerator {
    int GetCount(out int count);
    int GetSession(int index, out IAudioSessionControl2 session);
}
[ComImport, Guid("bfb7ff88-7239-4fc9-8fa2-07c950be9c6d"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionControl2 {
    int GetState(out int s);
    int GetDisplayName([MarshalAs(UnmanagedType.LPWStr)] out string n);
    int SetDisplayName([MarshalAs(UnmanagedType.LPWStr)] string n, ref Guid g);
    int GetIconPath([MarshalAs(UnmanagedType.LPWStr)] out string p);
    int SetIconPath([MarshalAs(UnmanagedType.LPWStr)] string p, ref Guid g);
    int GetGroupingParam(out Guid g);
    int SetGroupingParam(ref Guid o, ref Guid g);
    int RegisterAudioSessionNotification(IntPtr n);
    int UnregisterAudioSessionNotification(IntPtr n);
    int GetSessionIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string id);
    int GetSessionInstanceIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string id);
    int GetProcessId(out uint pid);
    int IsSystemSoundsSession();
    int SetDuckingPreference(bool optOut);
}
[ComImport, Guid("87CE5498-68D6-44E5-9215-6DA47EF883D7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface ISimpleAudioVolume {
    int SetMasterVolume(float level, ref Guid ctx);
    int GetMasterVolume(out float level);
    int SetMute(bool mute, ref Guid ctx);
    int GetMute(out bool mute);
}
[ComImport, Guid("C02216F6-8C67-4B5B-9D00-D008E73E0064"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioMeterInformation {
    int GetPeakValue(out float peak);
}

public static class GameAudio {
    public static List<string> Apply(HashSet<uint> pids, bool mute) {
        var result = new List<string>();
        var en = (IMMDeviceEnumerator)(new MMDeviceEnumeratorCom());
        IMMDevice dev; en.GetDefaultAudioEndpoint(0, 1, out dev);   // render, multimedia
        Guid iid = typeof(IAudioSessionManager2).GUID; object o;
        dev.Activate(ref iid, 23, IntPtr.Zero, out o);
        var mgr = (IAudioSessionManager2)o;
        IAudioSessionEnumerator se; mgr.GetSessionEnumerator(out se);
        int n; se.GetCount(out n);
        Guid ctx = Guid.Empty;
        for (int i = 0; i < n; i++) {
            IAudioSessionControl2 c; se.GetSession(i, out c);
            uint pid; c.GetProcessId(out pid);
            if (!pids.Contains(pid)) continue;
            float peak = -1; bool wasMuted = false; string how = "";
            try { ((IAudioMeterInformation)c).GetPeakValue(out peak); } catch (Exception ex) { how += " meter:" + ex.GetType().Name; }
            try {
                var vol = (ISimpleAudioVolume)c;
                try { vol.GetMute(out wasMuted); } catch {}
                vol.SetMute(mute, ref ctx);
                how += " mute:ok";
            } catch (Exception ex) { how += " mute:FAILED(" + ex.GetType().Name + ")"; }
            result.Add("pid " + pid + ": peak " + peak.ToString("0.000") + ", wasMuted=" + wasMuted + " -> muted=" + mute + " [" + how.Trim() + "]");
        }
        return result;
    }
}
"@

$pids = New-Object 'System.Collections.Generic.HashSet[uint32]'
Get-Process "Mini Motorways" -ErrorAction SilentlyContinue | ForEach-Object { [void]$pids.Add([uint32]$_.Id) }
"game processes: $($pids.Count)"
$out = [GameAudio]::Apply($pids, (-not $Unmute))
if ($out.Count -eq 0) { "no active audio sessions found for the game processes (nothing is currently producing/holding audio)" } else { $out }
