﻿//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using System;
using System.Runtime.Serialization;
using CNTK;

namespace EasyCNTK.Layers
{
    [Serializable]
    public abstract class Layer : ISerializable
    {
        public bool IsRecurrent { get; set; } = false;
        public abstract Function Create(Function input, DeviceDescriptor device);
        public abstract string GetDescription();

        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
        }
    }
}
