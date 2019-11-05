//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    /// <summary>
    /// The base class for creating teachers, contains the definition of the environment and auxiliary methods for calculating intermediate results
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class AgentTeacher<T> : IDisposable where T : IConvertible
    {
        protected Environment Environment { get; set; }
        protected DeviceDescriptor Device { get; set; }
        protected T[] Multiply(T[] vector, T factor)
        {
            var type = typeof(T);
            T[] result = new T[vector.Length];
            for (int i = 0; i < result.Length; i++)
            {
                double v = vector[i].ToDouble(CultureInfo.InvariantCulture);
                double f = factor.ToDouble(CultureInfo.InvariantCulture);
                result[i] = (T)Convert.ChangeType(v * f, type);
            }
            return result;
        }
        protected virtual T CalculateDiscountedReward(T[] rewards, double gamma)
        {
            var type = typeof(T);
            double totalReward = rewards[0].ToDouble(CultureInfo.InvariantCulture);
            for (int i = 1; i < rewards.Length; i++)
            {
                totalReward += rewards[i].ToDouble(CultureInfo.InvariantCulture) * Math.Pow(gamma, i);
            }
            return (T)Convert.ChangeType(totalReward, type);
        }
        public Func<Loss[]> GetLoss { get; set; } = () => new[] { new SquaredError() };
        public Func<Loss[]> GetEvalLoss { get; set; } = () => new[] { new SquaredError() };
        public Func<Optimizer[]> GetOptimizer { get; set; } = () => new Optimizer[] { new Adam(0.05, 0.9) };

        public AgentTeacher(Environment environment, DeviceDescriptor device)
        {
            Environment = environment;  
            Device = device;
        }

        #region IDisposable Support
        private bool disposedValue = false; // To identify redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    //Environment.Dispose();                    
                    //Device.Dispose();
                }

                // TODO: Release unmanaged resources (unmanaged objects) and override the completion method below.
                // TODO: set large fields to NULL.
                Environment = null;               
                Device = null;

                disposedValue = true;
            }
        }

        // TODO: Override the completion method only if Dispose (bool disposing) above includes code to free unmanaged resources.
        ~AgentTeacher()
        {
            // Do not modify this code. Place the cleanup code above in the Dispose (bool disposing) method.
            Dispose(false);
        }

        // This code has been added to properly implement the released class template.
        public void Dispose()
        {
            // Do not modify this code. Place the cleanup code above in the Dispose (bool disposing) method.
            Dispose(true);
            // TODO: uncomment the next line if the completion method is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
