using System.Collections.Generic;

namespace SinusoidRegressionLSTM
{
    static class HelperExtesions
    {
        /// <summary>
        /// Splits an input sequence into segments (subsequences) of equal size
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">Source sequence</param>
        /// <param name="segmentSize">Segment size (number of elements)</param>
        /// <returns></returns>
        public static IEnumerable<IList<T>> Segment<T>(this IEnumerable<T> source, int segmentSize)
        {
            IList<T> list = new List<T>(segmentSize);
            foreach (var item in source)
            {
                list.Add(item);
                if (list.Count == segmentSize)
                {
                    yield return list;
                    list = new List<T>(segmentSize);
                }
            }
            if (list.Count > 0)
            {
                yield return list;
            }
        }
    }
}
