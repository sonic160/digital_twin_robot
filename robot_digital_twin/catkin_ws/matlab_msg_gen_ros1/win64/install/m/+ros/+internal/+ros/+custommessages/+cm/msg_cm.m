function [data, info] = msg_cm
%msg_cm gives an empty data for cm/msg_cm
% Copyright 2019-2020 The MathWorks, Inc.
%#codegen
data = struct();
data.MessageType = 'cm/msg_cm';
[data.Header, info.Header] = ros.internal.ros.messages.std_msgs.header;
info.Header.MLdataType = 'struct';
[data.Name, info.Name] = ros.internal.ros.messages.ros.char('string',NaN);
[data.Position, info.Position] = ros.internal.ros.messages.ros.default_type('double',NaN);
[data.Temperature, info.Temperature] = ros.internal.ros.messages.ros.default_type('double',NaN);
[data.Voltage, info.Voltage] = ros.internal.ros.messages.ros.default_type('double',NaN);
info.MessageType = 'cm/msg_cm';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,10);
info.MatPath{1} = 'header';
info.MatPath{2} = 'header.seq';
info.MatPath{3} = 'header.stamp';
info.MatPath{4} = 'header.stamp.sec';
info.MatPath{5} = 'header.stamp.nsec';
info.MatPath{6} = 'header.frame_id';
info.MatPath{7} = 'name';
info.MatPath{8} = 'position';
info.MatPath{9} = 'temperature';
info.MatPath{10} = 'voltage';
