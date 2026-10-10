using System;
using System.IO;
using JetBrains.Annotations;

namespace Factory;

public interface IScope
{
	Assembler Assembler { get; }

	IScope ParentScope { get; }

	bool Release();

	void AddChildScope(IScope childScope, object establishingObject);

	object Get(Type type);

	T Get<T>() where T : class;

	void Assemble([NotNull] object unboundObject);

	bool Release(object obj);

	void Set(Type type, object variable);

	void Set<T>(object variable);

	void Unset(Type type);

	T Import<T>(BinaryReader reader) where T : class;

	object Import(BinaryReader reader);

	bool Export(object obj, BinaryWriter writer);

	void Subscribe(IScopeObserver newObserver);

	void Unsubscribe(IScopeObserver oldObserver);
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
