﻿//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using CNTK;

namespace EasyCNTK.ActivationFunctions
{
    public abstract class ActivationFunction
    {
        public abstract Function ApplyActivationFunction(Function variable, DeviceDescriptor device);
        public abstract string GetDescription();
    }

}
