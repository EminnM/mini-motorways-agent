using Mono.Cecil;
using Mono.Cecil.Cil;

// Surgical patch: add the missing `Key` property to System.Linq.IGrouping<TKey,TElement>
// and its concrete implementation System.Linq.Grouping<TKey,TElement>, in a COPY of
// System.Core.dll. The interface's Key getter and the class's implicit implementation
// were both stripped, but the class's own backing field `_key` (set elsewhere in the
// still-present GroupBy machinery) survives -- so this is a genuine restoration of a
// stripped accessor, not new behavior.

if (args.Length != 2)
{
    Console.Error.WriteLine("Usage: systemcore_patcher <input System.Core.dll> <output System.Core.dll>");
    return 1;
}

string inputPath = args[0];
string outputPath = args[1];

using var assembly = AssemblyDefinition.ReadAssembly(inputPath, new ReaderParameters { ReadWrite = false });
var module = assembly.MainModule;

TypeDefinition? igroupingType = module.GetType("System.Linq.IGrouping`2");
TypeDefinition? groupingType = module.GetType("System.Linq.Grouping`2");
if (igroupingType == null || groupingType == null)
{
    Console.Error.WriteLine($"FATAL: IGrouping`2 found={igroupingType != null}, Grouping`2 found={groupingType != null}.");
    return 1;
}

bool interfaceHasKey = igroupingType.Properties.Any(p => p.Name == "Key");
bool classHasKey = groupingType.Properties.Any(p => p.Name == "Key") || groupingType.Methods.Any(m => m.Name == "get_Key");

Console.WriteLine("Pre-patch verification:");
Console.WriteLine($"  IGrouping<TKey,TElement> found: yes");
Console.WriteLine($"  IGrouping.Key present already: {interfaceHasKey}");
Console.WriteLine($"  Grouping`2 found: yes");
Console.WriteLine($"  Grouping._key backing field present: {groupingType.Fields.Any(f => f.Name == "_key")}");
Console.WriteLine($"  Grouping.Key present already: {classHasKey}");
Console.WriteLine();

if (interfaceHasKey && classHasKey)
{
    Console.Error.WriteLine("Both Key members already exist -- no patch needed. Aborting to avoid duplicates.");
    return 1;
}

FieldDefinition? keyField = groupingType.Fields.FirstOrDefault(f => f.Name == "_key");
if (keyField == null)
{
    Console.Error.WriteLine("FATAL: Grouping._key backing field not found -- cannot build the patch on top of it.");
    return 1;
}

// The interface's own generic parameter (TKey of IGrouping<TKey, TElement>).
var interfaceTKey = igroupingType.GenericParameters.First(p => p.Name == "TKey");

if (!interfaceHasKey)
{
    var getKeyOnInterface = new MethodDefinition(
        "get_Key",
        MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.NewSlot |
        MethodAttributes.Abstract | MethodAttributes.Virtual | MethodAttributes.SpecialName,
        interfaceTKey);
    igroupingType.Methods.Add(getKeyOnInterface);

    var keyPropertyOnInterface = new PropertyDefinition("Key", PropertyAttributes.None, interfaceTKey)
    {
        GetMethod = getKeyOnInterface
    };
    igroupingType.Properties.Add(keyPropertyOnInterface);
    Console.WriteLine("Added IGrouping<TKey,TElement>.Key (abstract getter) to the interface.");
}

if (!classHasKey)
{
    // The class's own generic parameter (TKey of Grouping<TKey, TElement>) -- distinct
    // Cecil object from the interface's TKey even though both are named "TKey".
    var classTKey = groupingType.GenericParameters.First(p => p.Name == "TKey");

    var getKeyOnClass = new MethodDefinition(
        "get_Key",
        MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.SpecialName,
        classTKey);

    var il = getKeyOnClass.Body.GetILProcessor();
    il.Emit(OpCodes.Ldarg_0);                 // this
    il.Emit(OpCodes.Ldfld, keyField);         // this._key
    il.Emit(OpCodes.Ret);

    groupingType.Methods.Add(getKeyOnClass);

    var keyPropertyOnClass = new PropertyDefinition("Key", PropertyAttributes.None, classTKey)
    {
        GetMethod = getKeyOnClass
    };
    groupingType.Properties.Add(keyPropertyOnClass);
    Console.WriteLine("Added Grouping<TKey,TElement>.Key (public getter returning the existing _key field) -- implicitly satisfies IGrouping<TKey,TElement>.Key by name+signature, same mechanism the C# compiler itself would use.");
}

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
assembly.Write(outputPath);

Console.WriteLine($"Patched assembly written to: {outputPath}");
Console.WriteLine("Only the Key accessor was added on each type; every existing member is untouched.");
return 0;
