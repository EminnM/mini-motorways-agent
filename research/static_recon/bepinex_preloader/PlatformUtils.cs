using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using MonoMod.Utils;

namespace BepInEx.Preloader;

internal static class PlatformUtils
{
	[StructLayout((LayoutKind)0, Pack = 1)]
	public struct utsname_osx
	{
		private const int osx_utslen = 256;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string sysname;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string nodename;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string release;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string version;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
		public string machine;
	}

	[StructLayout((LayoutKind)0, Pack = 1)]
	public struct utsname_linux
	{
		private const int linux_utslen = 65;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string sysname;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string nodename;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string release;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string version;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string machine;

		[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 65)]
		public string domainname;
	}

	[DllImport("libc.so.6", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "uname")]
	private static extern IntPtr uname_linux(ref utsname_linux utsname);

	[DllImport("/usr/lib/libSystem.dylib", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, EntryPoint = "uname")]
	private static extern IntPtr uname_osx(ref utsname_osx utsname);

	/// <summary>
	/// Recreation of MonoMod's PlatformHelper.DeterminePlatform method, but with libc calls instead of creating processes.
	/// </summary>
	public static void SetPlatform()
	{
		Platform platform = Platform.Unknown;
		PropertyInfo property = typeof(Environment).GetProperty("Platform", BindingFlags.Static | BindingFlags.NonPublic);
		string text = (((object)property == null) ? Environment.OSVersion.Platform.ToString() : property.GetValue(null, new object[0]).ToString());
		text = text.ToLowerInvariant();
		if (text.Contains("win"))
		{
			platform = Platform.Windows;
		}
		else if (text.Contains("mac") || text.Contains("osx"))
		{
			platform = Platform.MacOS;
		}
		else if (text.Contains("lin") || text.Contains("unix"))
		{
			platform = Platform.Linux;
		}
		if (Is(platform, Platform.Linux) && Directory.Exists("/data") && File.Exists("/system/build.prop"))
		{
			platform = Platform.Android;
		}
		else if (Is(platform, Platform.Unix) && Directory.Exists("/System/Library/AccessibilityBundles"))
		{
			platform = Platform.iOS;
		}
		MethodInfo methodInfo = typeof(Environment).GetProperty("Is64BitOperatingSystem")?.GetGetMethod();
		platform = (Platform)(((object)methodInfo == null) ? ((int)platform | ((IntPtr.Size >= 8) ? 2 : 0)) : ((int)platform | (((bool)methodInfo.Invoke(null, new object[0])) ? 2 : 0)));
		if ((Is(platform, Platform.MacOS) || Is(platform, Platform.Linux)) && (object)Type.GetType("Mono.Runtime") != null)
		{
			IntPtr intPtr;
			string machine;
			if (Is(platform, Platform.MacOS))
			{
				utsname_osx utsname = default(utsname_osx);
				intPtr = uname_osx(ref utsname);
				machine = utsname.machine;
			}
			else
			{
				utsname_linux utsname2 = default(utsname_linux);
				intPtr = uname_linux(ref utsname2);
				machine = utsname2.machine;
			}
			if (intPtr == IntPtr.Zero && (machine.StartsWith("aarch") || machine.StartsWith("arm")))
			{
				platform |= Platform.ARM;
			}
		}
		else
		{
			PortableExecutableKinds portableExecutableKinds = default(PortableExecutableKinds);
			ImageFileMachine imageFileMachine = default(ImageFileMachine);
			typeof(object).Module.GetPEKind(ref portableExecutableKinds, ref imageFileMachine);
			if (imageFileMachine == ImageFileMachine.ARM)
			{
				platform |= Platform.ARM;
			}
		}
		PlatformHelper.Current = platform;
	}

	private static bool Is(Platform current, Platform expected)
	{
		return (current & expected) == expected;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
