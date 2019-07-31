using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Saigo.Core.Graphs
{
    public class TreeNode<T> : IEnumerable<TreeNode<T>>
    {
        private readonly LinkedList<TreeNode<T>> _children;

        public T Data { get; set; }
        public TreeNode<T> Parent { get; set; }

        public IReadOnlyCollection<TreeNode<T>> Children => new ReadOnlyCollection<TreeNode<T>>(_children.ToList());

        public bool IsRoot => Parent == null;

        public bool IsLeaf => Children.Count == 0;

        public int Level
        {
            get
            {
                if (IsRoot)
                    return 0;
                return Parent.Level + 1;
            }
        }

        public TreeNode(T data)
        {
            this.Data = data;
            _children = new LinkedList<TreeNode<T>>();

            this.ElementsIndex = new LinkedList<TreeNode<T>>();
            this.ElementsIndex.Add(this);
        }

        public TreeNode<T> AddChild(T child)
        {
            var childNode = new TreeNode<T>(child) { Parent = this };

            _children.AddLast(childNode);

            this.RegisterChildForSearch(childNode);

            return childNode;
        }

        public override string ToString()
        {
            return Data != null ? Data.ToString() : "[data null]";
        }


        #region searching

        private ICollection<TreeNode<T>> ElementsIndex { get; set; }

        private void RegisterChildForSearch(TreeNode<T> node)
        {
            ElementsIndex.Add(node);
            Parent?.RegisterChildForSearch(node);
        }

        public TreeNode<T> FindChild(Func<TreeNode<T>, bool> predicate)
        {
            return this.ElementsIndex.FirstOrDefault(predicate);
        }

        #endregion


        #region iterating

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public IEnumerator<TreeNode<T>> GetEnumerator()
        {
            yield return this;
            foreach (var directChild in this.Children)
            {
                foreach (var anyChild in directChild)
                    yield return anyChild;
            }
        }

        #endregion
    }
}
