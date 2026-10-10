using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using MelonLoader;

[assembly: MelonInfo(typeof(MiniMotorwaysMelonHelloWorld.Plugin), "MiniMotorwaysMelonHelloWorld", "0.1.0", "local-research")]
[assembly: MelonPlatform(MelonPlatformAttribute.CompatiblePlatforms.WINDOWS_X86)]

namespace MiniMotorwaysMelonHelloWorld
{
    public class Plugin : MelonPlugin
    {
        // Dedicated log file, independent of MelonLoader's own Latest.log,
        // written into the research workspace (not the Steam install dir),
        // matching the same pattern used for the BepInEx hello-world plugin.
        private static readonly string LogPath =
            @"C:\Users\TKA\Desktop\mini moto\logs\instrumentation.log";

        private static void WriteLine(string message)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(LogPath));
                string line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {message}";
                File.AppendAllText(LogPath, line + Environment.NewLine);
            }
            catch
            {
                // Never let logging failures affect the game itself.
            }
        }

        public override void OnInitializeMelon()
        {
            try
            {
                int pid = Process.GetCurrentProcess().Id;
                string unityVersion = MelonLoader.InternalUtils.UnityInformationHandler.EngineVersion.ToString();
                string gameName = MelonLoader.InternalUtils.UnityInformationHandler.GameName;
                string gameVersion = MelonLoader.InternalUtils.UnityInformationHandler.GameVersion;

                Assembly appAssembly = null;
                string appAssemblyInfo = "UNKNOWN";
                try
                {
                    appAssembly = Assembly.Load("App");
                    appAssemblyInfo = $"{appAssembly.FullName} (location={appAssembly.Location})";
                }
                catch (Exception ex)
                {
                    appAssemblyInfo = $"FAILED TO LOAD: {ex.Message}";
                }

                WriteLine("=== MiniMotorwaysMelonHelloWorld plugin initializing (MelonLoader) ===");
                WriteLine("[MiniMotorwaysAI] Plugin loaded successfully");
                WriteLine($"Process ID: {pid}");
                WriteLine($"Unity version (MelonUtils.GetUnityVersion): {unityVersion}");
                WriteLine($"Game: {gameName} {gameVersion}");
                WriteLine($"App.dll assembly info: {appAssemblyInfo}");
                WriteLine("=== Initialization complete ===");

                LoggerInstance.Msg("[MiniMotorwaysAI] Plugin loaded successfully");
                LoggerInstance.Msg($"PID={pid}, Unity={unityVersion}. See instrumentation_logs/instrumentation.log for details.");
            }
            catch (Exception ex)
            {
                WriteLine($"EXCEPTION during OnInitializeMelon: {ex}");
                LoggerInstance.Error($"Exception during OnInitializeMelon: {ex}");
            }
        }
    }
}
