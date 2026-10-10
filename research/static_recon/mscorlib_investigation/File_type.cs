using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace System.IO;

public static class File
{
	public static StreamReader OpenText(string path)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path");
		}
		return new StreamReader(path);
	}

	public static StreamWriter CreateText(string path)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path");
		}
		return new StreamWriter(path, append: false);
	}

	public static void Copy(string sourceFileName, string destFileName)
	{
		Copy(sourceFileName, destFileName, overwrite: false);
	}

	public static void Copy(string sourceFileName, string destFileName, bool overwrite)
	{
		if (sourceFileName == null)
		{
			throw new ArgumentNullException("sourceFileName", "File name cannot be null.");
		}
		if (destFileName == null)
		{
			throw new ArgumentNullException("destFileName", "File name cannot be null.");
		}
		if (sourceFileName.Length == 0)
		{
			throw new ArgumentException("Empty file name is not legal.", "sourceFileName");
		}
		if (destFileName.Length == 0)
		{
			throw new ArgumentException("Empty file name is not legal.", "destFileName");
		}
		FileSystem.CopyFile(Path.GetFullPath(sourceFileName), Path.GetFullPath(destFileName), overwrite);
	}

	public static FileStream Create(string path)
	{
		return Create(path, 4096);
	}

	public static FileStream Create(string path, int bufferSize)
	{
		return new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None, bufferSize);
	}

	public static void Delete(string path)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path");
		}
		FileSystem.DeleteFile(Path.GetFullPath(path));
	}

	public static bool Exists(string path)
	{
		try
		{
			if (path == null)
			{
				return false;
			}
			if (path.Length == 0)
			{
				return false;
			}
			path = Path.GetFullPath(path);
			if (path.Length > 0 && PathInternal.IsDirectorySeparator(path[path.Length - 1]))
			{
				return false;
			}
			return FileSystem.FileExists(path);
		}
		catch (ArgumentException)
		{
		}
		catch (IOException)
		{
		}
		catch (UnauthorizedAccessException)
		{
		}
		return false;
	}

	public static FileStream Open(string path, FileMode mode)
	{
		return Open(path, mode, (mode == FileMode.Append) ? FileAccess.Write : FileAccess.ReadWrite, FileShare.None);
	}

	public static FileStream Open(string path, FileMode mode, FileAccess access)
	{
		return Open(path, mode, access, FileShare.None);
	}

	public static FileStream Open(string path, FileMode mode, FileAccess access, FileShare share)
	{
		return new FileStream(path, mode, access, share);
	}

	public static DateTime GetCreationTime(string path)
	{
		return FileSystem.GetCreationTime(Path.GetFullPath(path)).LocalDateTime;
	}

	public static DateTime GetLastWriteTime(string path)
	{
		return FileSystem.GetLastWriteTime(Path.GetFullPath(path)).LocalDateTime;
	}

	public static DateTime GetLastWriteTimeUtc(string path)
	{
		return FileSystem.GetLastWriteTime(Path.GetFullPath(path)).UtcDateTime;
	}

	public static FileAttributes GetAttributes(string path)
	{
		return FileSystem.GetAttributes(Path.GetFullPath(path));
	}

	public static void SetAttributes(string path, FileAttributes fileAttributes)
	{
		if ((fileAttributes & (FileAttributes)(-2147483648)) != 0)
		{
			Path.Validate(path);
			if (!MonoIO.SetFileAttributes(path, fileAttributes, out var error))
			{
				throw MonoIO.GetException(path, error);
			}
		}
		else
		{
			FileSystem.SetAttributes(Path.GetFullPath(path), fileAttributes);
		}
	}

	public static FileStream OpenRead(string path)
	{
		return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
	}

	public static FileStream OpenWrite(string path)
	{
		return new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
	}

	public static string ReadAllText(string path)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path");
		}
		if (path.Length == 0)
		{
			throw new ArgumentException("Empty path name is not legal.", "path");
		}
		return InternalReadAllText(path, Encoding.UTF8);
	}

	private static string InternalReadAllText(string path, Encoding encoding)
	{
		using StreamReader streamReader = new StreamReader(path, encoding, detectEncodingFromByteOrderMarks: true);
		return streamReader.ReadToEnd();
	}

	public static byte[] ReadAllBytes(string path)
	{
		using FileStream fileStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1);
		long length = fileStream.Length;
		if (length > int.MaxValue)
		{
			throw new IOException("The file is too long. This operation is currently limited to supporting files less than 2 gigabytes in size.");
		}
		if (length == 0L)
		{
			return ReadAllBytesUnknownLength(fileStream);
		}
		int num = 0;
		int num2 = (int)length;
		byte[] array = new byte[num2];
		while (num2 > 0)
		{
			int num3 = fileStream.Read(array, num, num2);
			if (num3 == 0)
			{
				throw Error.GetEndOfFile();
			}
			num += num3;
			num2 -= num3;
		}
		return array;
	}

	private static byte[] ReadAllBytesUnknownLength(FileStream fs)
	{
		byte[] array = null;
		Span<byte> span = stackalloc byte[512];
		try
		{
			int num = 0;
			while (true)
			{
				if (num == span.Length)
				{
					uint num2 = (uint)(span.Length * 2);
					if (num2 > 2147483591)
					{
						num2 = (uint)Math.Max(2147483591, span.Length + 1);
					}
					byte[] array2 = ArrayPool<byte>.Shared.Rent((int)num2);
					span.CopyTo(array2);
					if (array != null)
					{
						ArrayPool<byte>.Shared.Return(array);
					}
					span = (array = array2);
				}
				int num3 = fs.Read(span.Slice(num));
				if (num3 == 0)
				{
					break;
				}
				num += num3;
			}
			return span.Slice(0, num).ToArray();
		}
		finally
		{
			if (array != null)
			{
				ArrayPool<byte>.Shared.Return(array);
			}
		}
	}

	public static void WriteAllBytes(string path, byte[] bytes)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path", "Path cannot be null.");
		}
		if (path.Length == 0)
		{
			throw new ArgumentException("Empty path name is not legal.", "path");
		}
		if (bytes == null)
		{
			throw new ArgumentNullException("bytes");
		}
		InternalWriteAllBytes(path, bytes);
	}

	private static void InternalWriteAllBytes(string path, byte[] bytes)
	{
		using FileStream fileStream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
		fileStream.Write(bytes, 0, bytes.Length);
	}

	public static void WriteAllLines(string path, string[] contents)
	{
		WriteAllLines(path, (IEnumerable<string>)contents);
	}

	public static void WriteAllLines(string path, IEnumerable<string> contents)
	{
		if (path == null)
		{
			throw new ArgumentNullException("path");
		}
		if (contents == null)
		{
			throw new ArgumentNullException("contents");
		}
		if (path.Length == 0)
		{
			throw new ArgumentException("Empty path name is not legal.", "path");
		}
		InternalWriteAllLines(new StreamWriter(path), contents);
	}

	private static void InternalWriteAllLines(TextWriter writer, IEnumerable<string> contents)
	{
		using (writer)
		{
			foreach (string content in contents)
			{
				writer.WriteLine(content);
			}
		}
	}

	public static void Move(string sourceFileName, string destFileName)
	{
		if (sourceFileName == null)
		{
			throw new ArgumentNullException("sourceFileName", "File name cannot be null.");
		}
		if (destFileName == null)
		{
			throw new ArgumentNullException("destFileName", "File name cannot be null.");
		}
		if (sourceFileName.Length == 0)
		{
			throw new ArgumentException("Empty file name is not legal.", "sourceFileName");
		}
		if (destFileName.Length == 0)
		{
			throw new ArgumentException("Empty file name is not legal.", "destFileName");
		}
		string fullPath = Path.GetFullPath(sourceFileName);
		string fullPath2 = Path.GetFullPath(destFileName);
		if (!FileSystem.FileExists(fullPath))
		{
			throw new FileNotFoundException(SR.Format("Could not find file '{0}'.", fullPath), fullPath);
		}
		FileSystem.MoveFile(fullPath, fullPath2);
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
