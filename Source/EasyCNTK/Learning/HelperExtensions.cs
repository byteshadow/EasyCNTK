//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace EasyCNTK.Learning
{
    public static class HelperExtensions
    {
        /// <summary>
        /// Shuffles data in a collection.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="seed">The initial value for the random number generator (<seealso cref="Random"/>), если 0 - используется генератор по умолчанию </param>
        public static void Shuffle<T>(this IList<T> source, int seed = 0)
        {
            Random random = seed== 0 ? new Random() : new Random(seed);
            int countLeft = source.Count;
            while (countLeft > 1)
            {
                countLeft--;
                int indexNextItem = random.Next(countLeft + 1);
                T temp = source[indexNextItem];
                source[indexNextItem] = source[countLeft];
                source[countLeft] = temp;
            }
        }
        /// <summary>
        /// Splits a data set into 2 parts in a given ratio
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">Source dataset</param>
        /// <param name="percent">The size of the first data set as a percentage of the original collection. Must be in the range [0; 1].</param>
        /// <param name="first">First data set</param>
        /// <param name="second">Second data set</param>
        /// <param name="randomizeSplit"&gt; Random partitioning (data for sets are taken randomly from the entire sample)</param>        
        /// <param name="seed">The initial value for the random number generator, if 0 - the default generator is used</param>
        public static void Split<T>(this IList<T> source, double percent, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            if (percent > 1 || percent < 0)
            {
                throw new ArgumentOutOfRangeException("Percent must be in range [0;1]", "percent");
            }
            int firstCount = (int)(source.Count * percent);

            if (randomizeSplit)
            {
                source.Shuffle(seed);
            }
            first = new List<T>(source.Take(firstCount));
            second = new List<T>(source.Skip(firstCount));
        }
        /// <summary>
        /// Splits a data set into 2 parts in a given ratio, keeping the original class distribution unchanged for both collections. Assumes that one example contains one class (The task of a single-class classification).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="source">Source dataset</param>
        /// <param name="percent">The size of the first data set as a percentage of the original collection. Must be in the range [0; 1].</param>
        /// <param name="labelSelector">Class label selector</param>
        /// <param name="labelComparer">Comparator, used to determine the equality of labels of two classes</param>
        /// <param name="first">First data set</param>
        /// <param name="second">Second data set</param>
        /// <param name="randomizeSplit">&gt; Random partitioning (data for sets are taken randomly from the entire sample)</param>
        /// <param name="seed">The initial value for the random number generator, if 0 - the default generator is used</param>
        public static void SplitBalanced<T, U>(this IList<T> source, double percent, Func<T, U> labelSelector, IEqualityComparer<U> labelComparer, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            if (percent > 1 || percent < 0)
            {
                throw new ArgumentOutOfRangeException("Percent must be in range [0;1]", "percent");
            }
            if (source.Count < 2)
            {
                throw new ArgumentOutOfRangeException("Count elements in source collection must be greater 1", "source");
            }

            int firstCount = (int)(source.Count * percent);
            first = new List<T>(firstCount);
            second = new List<T>(source.Count - firstCount);

            if (randomizeSplit)
            {
                source.Shuffle(seed);
            }

            var groupedByLabel = labelComparer == null 
                ? source.GroupBy(labelSelector) 
                : source.GroupBy(labelSelector, labelComparer);
            foreach (var labelGroup in groupedByLabel)
            {
                var labelCount = labelGroup.Count();
                int toFirst = (int)(labelCount * percent);               

                foreach (var item in labelGroup)
                {
                    if (toFirst != 0)
                    {
                        first.Add(item);
                        toFirst--;
                        continue;
                    }
                    second.Add(item);
                }                
            }
        }

        /// <summary>
        /// Splits a data set into 2 parts in a given ratio, keeping the original class distribution unchanged for both collections.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="U"></typeparam>
        /// <param name="source">Source dataset</param>
        /// <param name="percent">The size of the first data set as a percentage of the original collection. Must be in the range [0; 1].</param>
        /// <param name="labelSelector">Class label selector</param>
        /// <param name="first">First data set</param>
        /// <param name="second">Second data set</param>
        /// <param name="randomizeSplit">&gt; Random partitioning (data for sets are taken randomly from the entire sample)</param>
        /// <param name="seed">The initial value for the random number generator, if 0 - the default generator is used</param>
        public static void SplitBalanced<T, U>(this IList<T> source, double percent, Func<T, U> labelSelector, out IList<T> first, out IList<T> second, bool randomizeSplit = false, int seed = 0)
        {
            source.SplitBalanced(percent, labelSelector, null, out first, out second, randomizeSplit, seed);
        }

        /// <summary>
        /// Based on a given set of examples of one class, creates the required number of synthetic examples of the same class using the Synthetic Minority Over-sampling Technique (SMOTE) method
        /// For correct generation, the input data must be normalized.
        /// </summary>
        /// <typeparam name="T">Type of elements; all types implementing<seealso cref="IConvertible"/></typeparam>
        /// <param name="source"></param>
        /// <param name="similarSampleCount">Number of similar synthetic examples to create</param>
        /// <param name="nearestNeighborsCount">The number of nearest neighbors used for the generation of a synthetic example</param>
        /// <param name="seed">The initial value for the random number generator, if 0 - the default generator is used</param>
        /// <returns></returns>
        public static IList<T[]> MakeSimilarSamplesBySMOTE<T>(this IList<T[]> source, int similarSampleCount, int nearestNeighborsCount = 5, int seed = 0) where T : IConvertible
        {
            double computeDistance(T[] first, T[] second)
            {
                if (first.Length != second.Length) throw new IndexOutOfRangeException("Размерность одного из примеров датасета отличается от остальных");
                double distance = 0;

                for (int i = 0; i < first.Length; i++)
                {
                    distance += Math.Abs(first[i].ToDouble(CultureInfo.InvariantCulture) - second[i].ToDouble(CultureInfo.InvariantCulture));
                }
                return distance;
            }

            if (source.Count < 2)
                throw new ArgumentException("Source collection должна содержать минимум 2 элемента.", "source");
            if (similarSampleCount < 1)
                throw new ArgumentOutOfRangeException("similarSampleCount", "Число синтезируемых примеров должно быть больше 0.");
            if (nearestNeighborsCount < 1)
                throw new ArgumentOutOfRangeException("nearestNeighborsCount", "Число соседей должны быть больше 0.");

            #region находим [nearestNeighborsCount] ближайших соседей для каждого примера
            var nearestNeighborsIndexes = source
                   .Select(p => Enumerable.Range(0, nearestNeighborsCount)
                       .Select(q => (index: -1, distance: double.MaxValue))
                       .ToList())
                   .ToList();

            for (int first = 0; first < source.Count; first++)
            {
                for (int second = first + 1; second < source.Count; second++)
                {
                    double distance = computeDistance(source[first], source[second]);

                    #region обновляем индексы ближайших [nearestNeighborsCount] соседей у текущего примера и сравниваемого
                    nearestNeighborsIndexes[first].Sort((a, b) =>
                    {
                        if (a.distance > b.distance)
                            return -1;
                        if (a.distance < b.distance)
                            return 1;
                        return 0;
                    }); //sorting current neighbors in descending order of distance
                    for (int k = 0; k < nearestNeighborsIndexes[first].Count; k++)
                    {
                        if (distance < nearestNeighborsIndexes[first][k].distance)
                        {
                            nearestNeighborsIndexes[first][k] = (second, distance);

                            //if the second (compared) neighbor has more distant neighbors, replace them with the first (current) neighbor
                            int indexWorstNeighbor = nearestNeighborsIndexes[second].FindIndex(p => p.distance > distance);
                            if (indexWorstNeighbor != -1)
                            {
                                nearestNeighborsIndexes[second][indexWorstNeighbor] = (first, distance);
                            }
                            break;
                        }
                    }
                    #endregion
                }
            }
            #endregion

            #region создаем заданное количество синтетических примеров
            Random rnd = seed == 0 ? new Random() : new Random(seed);
            var elementTypeCode = source[0][0].GetTypeCode();
            var result = new List<T[]>(similarSampleCount);
            for (int i = 0; i < similarSampleCount; i++)
            {
                int indexRealSample = rnd.Next(source.Count);
                var neighborIndexes = nearestNeighborsIndexes[indexRealSample]
                    .Where(p => p.index != -1)
                    .ToList();
                int indexRealNeighbor = neighborIndexes[rnd.Next(0, neighborIndexes.Count)].index;

                var syntheticSample = new T[source[indexRealSample].Length];
                for (int element = 0; element < syntheticSample.Length; element++)
                {
                    double difference = source[indexRealSample][element].ToDouble(CultureInfo.InvariantCulture)
                                                - source[indexRealNeighbor][element].ToDouble(CultureInfo.InvariantCulture);
                    double gap = rnd.NextDouble();
                    syntheticSample[element] = (T)Convert.ChangeType(gap * difference + source[indexRealNeighbor][element].ToDouble(CultureInfo.InvariantCulture), elementTypeCode);
                }
                result.Add(syntheticSample);
            }
            #endregion

            return result;
        }

        /// <summary>
        /// Calculates statistics for each item in the collection. Allows loss of accuracy when calculating values out of range<seealso cref="double"/>
        /// </summary>
        /// <typeparam name="T">It is supported:<seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/></typeparam>
        /// <param name="source">Data set. Arrays with the same length</param>
        /// <param name="epsilon">The difference by which 2 floating-point numbers must be different in order to be considered different</param>
        /// <returns></returns>
        public static List<FeatureStatistic> ComputeStatisticForCollection<T>(this IEnumerable<IList<T>> source, double epsilon = 0.5) where T: IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement == null)
            {
                return new List<FeatureStatistic>();
            }            
            var result = firstElement
                .Select((p, i) => new FeatureStatistic(epsilon)
                {
                    FeatureName = (i + 1).ToString(),
                    Min = double.MaxValue,
                    Max = double.MinValue
                })
                .ToList();

            int countItems = 0;
            foreach (var item in source)
            {
                countItems++;
                for (int i = 0; i < item.Count; i++)
                {
                    double value = item[i].ToDouble(CultureInfo.InvariantCulture);

                    if (value < result[i].Min)
                    {
                        result[i].Min = value;
                    }
                    if (value > result[i].Max)
                    {
                        result[i].Max = value;
                    }

                    result[i].Average += value;

                    if (result[i].UniqueValues.TryGetValue(value, out int count))
                    {
                        result[i].UniqueValues[value]++;
                    }
                    else
                    {
                        result[i].UniqueValues[value] = 1;
                    }

                }
            }

            result.ForEach(p =>
            {
                p.Average = p.Average / countItems;

                p.MeanAbsoluteDeviation = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Abs(z.Key - p.Average) * z.Value) / countItems;

                p.Variance = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Pow(z.Key - p.Average, 2) * z.Value) / countItems;

                p.StandardDeviation = Math.Sqrt(p.Variance);

                #region поиск медианы
                int halfItems = countItems / 2; //half records
                int countElements = 0; //cumulative number of elements when searching for the median

                for (int i = 0; i < p.UniqueValues.Count; i++)
                {
                    countElements += p.UniqueValues.Values[i];
                    if (countItems % 2 == 0) //even number of elements
                    {
                        if (countElements == halfItems) // 122|345
                        {
                            p.Median = (p.UniqueValues.Keys[i] + p.UniqueValues.Keys[i + 1]) / 2;
                            break;
                        }
                        else if (countElements > halfItems) //122|225
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                    else
                    {
                        if (countElements >= halfItems + 1) //12 | 2 | 34 or 12 | 2 | 24 PS +1 because halfItems when dividing int / int will be one less than the actual number of elements
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                }
                #endregion

            });

            return result;
        }
        /// <summary>
        /// Computes object statistics for each type property:<seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/>. Допускает потерю точности при вычислении значений вне диапазона <seealso cref="double"/>
        /// </summary>
        /// <typeparam name="TModel"></typeparam>
        /// <param name="source">Data set</param>
        /// <param name="withoutProperties">Properties for which you do not need to calculate statistics</param>
        /// <param name="epsilon">The difference by which 2 floating-point numbers must be different in order to be considered different</param>
        /// <returns></returns>
        public static List<FeatureStatistic> ComputeStatisticForObject<TModel>(this IEnumerable<TModel> source, double epsilon = 0.5, string[] withoutProperties = null) where TModel : class
        {
            if (withoutProperties == null)
            {
                withoutProperties = new string[0];
            }
            var supportedTypes = new Type[] { typeof(int), typeof(long), typeof(float), typeof(double), typeof(decimal) };

            var firstElement = source.FirstOrDefault();
            if (firstElement == null)
            {
                return new List<FeatureStatistic>();
            }

            var properties = firstElement.GetType()
                .GetProperties()
                .Where(p => !withoutProperties.Contains(p.Name))
                .Where(p => supportedTypes.Contains(p.PropertyType))
                .OrderBy(p => p.Name)
                .ToList();

            var result = properties
                .Select(p => new FeatureStatistic(epsilon)
                {
                    FeatureName = p.Name,
                    Min = double.MaxValue,
                    Max = double.MinValue
                })
                .ToList();

            int countItems = 0;
            foreach (var item in source)
            {
                countItems++;
                for (int i = 0; i < properties.Count; i++)
                {
                    double value = Convert.ToDouble(properties[i].GetValue(item));

                    if (value < result[i].Min)
                    {
                        result[i].Min = value;
                    }
                    if (value > result[i].Max)
                    {
                        result[i].Max = value;
                    }

                    result[i].Average += value;

                    if (result[i].UniqueValues.TryGetValue(value, out int count))
                    {
                        result[i].UniqueValues[value]++;
                    }
                    else
                    {
                        result[i].UniqueValues[value] = 1;
                    }

                }
            }

            result.ForEach(p =>
            {
                p.Average = p.Average / countItems;

                p.MeanAbsoluteDeviation = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Abs(z.Key - p.Average) * z.Value) / countItems;

                p.Variance = p.UniqueValues.Aggregate(0.0, (sum, z) => sum + Math.Pow(z.Key - p.Average, 2) * z.Value) / countItems;

                p.StandardDeviation = Math.Sqrt(p.Variance);

                #region поиск медианы
                int halfItems = countItems / 2; //half records
                int countElements = 0; //cumulative number of elements when searching for the median

                for (int i = 0; i < p.UniqueValues.Count; i++)
                {
                    countElements += p.UniqueValues.Values[i];
                    if (countItems % 2 == 0) //even number of elements
                    {
                        if (countElements == halfItems) // 122|345
                        {
                            p.Median = (p.UniqueValues.Keys[i] + p.UniqueValues.Keys[i + 1]) / 2;
                            break;
                        }
                        else if (countElements > halfItems) //122|225
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                    else
                    {
                        if (countElements >= halfItems + 1) //12 | 2 | 34 or 12 | 2 | 24 PS +1 because halfItems when dividing int / int will be one less than the actual number of elements
                        {
                            p.Median = p.UniqueValues.Keys[i];
                            break;
                        }
                    }
                }
                #endregion

            });

            return result;
        }
        
        public static IList<double[]> MinMaxNormalize(this IList<double[]> source, bool centerOnXaxis, out double[] mins, out double[] maxes)
        {
            mins = source[0].Select(p => double.MaxValue).ToArray();
            maxes = source[0].Select(p => double.MinValue).ToArray();

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Length; j++)
                {
                    if (source[i][j] > maxes[j])
                    {
                        maxes[j] = source[i][j];
                    }
                    if (source[i][j] < mins[j])
                    {
                        mins[j] = source[i][j];
                    }
                }
            }

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Length; j++)
                {
                    var toZeroPoint = source[i][j] - mins[j];
                    if (centerOnXaxis)
                    {
                        source[i][j] = (toZeroPoint / Math.Abs(maxes[j] - mins[j]) - 0.5) * 2;
                    }
                    else
                    {
                        source[i][j] = toZeroPoint / Math.Abs(maxes[j] - mins[j]);
                    }
                }
            }
            return source;
        }
        /// <summary>
        /// Performs normalization of each selected property by the formula: Xnorm = X / (| Xmax-Xmin |) (brings Xnorm to the range [0; 1]). Changes affect the original collection. Supported property types:<seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/>
        /// </summary>
        /// <typeparam name="TModel">The type of class whose properties you want to normalize</typeparam>
        /// <param name="source">Source collection</param>
        /// <param name="modelMaxes">An instance of a type that is initialized by the maxima of the corresponding values</param>
        /// <param name="modelMins">An instance of a type that is initialized with the minima of the corresponding values</param>
        /// <param name="propertyNames">The names of the properties for which normalization is required. If null or empty, performs normalization for all supported properties</param>
        /// <param name="centerOnXaxis">Indicates whether to normalize a value to the range [-1; one]. If the value of an element from the original collection is more / less than the maximum / minimum by the specified corresponding parameters, then it is possible to go beyond the range [-1; 1]</param>
        /// <returns></returns>
        public static IList<TModel> MinMaxNormalize<TModel>(this IList<TModel> source, TModel modelMaxes, TModel modelMins, bool centerOnXaxis, IList<string> propertyNames = null) where TModel : class
        {
            if (propertyNames == null || propertyNames.Count == 0)
            {
                var supportedTypes = new Type[] { typeof(int), typeof(long), typeof(float), typeof(double), typeof(decimal) };
                propertyNames = typeof(TModel)
                    .GetProperties()
                    .Where(p => supportedTypes.Contains(p.PropertyType))
                    .Select(p => p.Name)
                    .ToList();
            }

            var properties = source[0]
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .ToList();
            var mins = modelMins?
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .Select(p => Convert.ToDouble(p.GetValue(modelMins)))
                .ToList();
            var maxes = modelMaxes?
                .GetType()
                .GetProperties()
                .Where(p => propertyNames.Contains(p.Name))
                .OrderBy(p => p.Name)
                .Select(p => Convert.ToDouble(p.GetValue(modelMaxes)))
                .ToList();

            if (mins == null || maxes == null)
            {
                mins = properties.Select(p => double.MaxValue).ToList();
                maxes = properties.Select(p => double.MinValue).ToList();

                for (int i = 0; i < source.Count; i++)
                {
                    for (int j = 0; j < mins.Count; j++)
                    {
                        dynamic itemValue = properties[j].GetValue(source[i]);
                        if (itemValue > maxes[j])
                        {
                            maxes[j] = itemValue;
                        }
                        if (itemValue < mins[j])
                        {
                            mins[j] = itemValue;
                        }
                    }
                }
            }

            for (int i = 0; i < source.Count; i++)
            {
                for (int j = 0; j < mins.Count; j++)
                {
                    dynamic itemValue = properties[j].GetValue(source[i]);
                    var toZeroPoint = itemValue - mins[j];
                    var normalValue = toZeroPoint / Math.Abs(maxes[j] - mins[j]);
                    if (centerOnXaxis)
                    {
                        normalValue = (toZeroPoint / Math.Abs(maxes[j] - mins[j]) - 0.5) * 2;
                    }
                    properties[j].SetValue(source[i], normalValue);
                }
            }
            return source;
        }
        /// <summary>
        /// Performs normalization of each selected property by the formula: Xnorm = X / (| Xmax-Xmin |) (brings Xnorm to the range [0; 1]). Changes affect the original collection. Supported property types:<seealso cref="int"/>, <seealso cref="long"/>, <seealso cref="float"/>, <seealso cref="double"/>, <seealso cref="decimal"/> 
        /// </summary>
        /// <typeparam name="TModel"></typeparam>
        /// <param name="source">Source collection</param>
        /// <param name="centerOnXaxis">Indicates whether to normalize a value to the range [-1; 1].</param>
        /// <param name="propertyNames">The names of the properties for which normalization is required. If null or empty, performs normalization for all supported properties</param>
        /// <returns></returns>
        public static IList<TModel> MinMaxNormalize<TModel>(this IList<TModel> source, bool centerOnXaxis, IList<string> propertyNames = null) where TModel : class
        {
            return source.MinMaxNormalize(null, null, centerOnXaxis, propertyNames);
        }

    }
}
