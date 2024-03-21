classdef Server < WebSocketServer

    methods
        function obj = Server(varargin)
            %Constructor
            obj@WebSocketServer(varargin{:});
            rosinit('192.168.0.103', 11311);
        end

        function delete(obj)    
            rosshutdown;
        end
    end
    
    methods (Access = protected)
        function onOpen(obj,conn,message)
            motorNames = {'motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6'};
            fprintf('%s\n',"testing");
            while true
                msg = rostopic("echo", "/condition_monitoring");
                time = datestr(datetime('now'), 'HH:MM:SS.FFF');
                positionMatch = regexp(msg.Data, 'position: \[([^\]]+)\]', 'tokens', 'once');
                temperatureMatch = regexp(msg.Data, 'temperature: \[([^\]]+)\]', 'tokens', 'once');
                voltageMatch = regexp(msg.Data, 'voltage: \[([^\]]+)\]', 'tokens', 'once');
                
                % 转换提取的字符串数据为数值数组
                positionData = rmmissing(str2double(strsplit(positionMatch{1}, ' ')));
                temperatureData = rmmissing(str2double(strsplit(temperatureMatch{1}, ' ')));
                voltageData = rmmissing(str2double(strsplit(voltageMatch{1}, ' ')));
                
                % 创建字典对象
                dataDict = struct('position', positionData, 'temperature', temperatureData, 'voltage', voltageData);

                position = dataDict.position;
                temperature = dataDict.temperature;
                voltage = dataDict.voltage;
                disp(position);
                disp(temperature);
                disp(voltage);
                output = containers.Map();
                for i = 1:length(motorNames)
                    tmap = containers.Map();
                    tmap("a") = position(i);
                    tmap("t") = temperature(i);
                    tmap("v") = voltage(i);
                    output(motorNames{i}) = tmap;
                end
                finalOut = containers.Map();
                finalOut("data") = output;
                finalOut("time") = time;
                jsonStr = jsonencode(finalOut);
                disp(jsonStr);
                conn.send(jsonStr);
                % pause(0.1);
            end
        end
        
        function onTextMessage(obj,conn,message)
            % This function sends an echo back to the client
            conn.send(message); % Echo
        end
        
        function onBinaryMessage(obj,conn,bytearray)
            % This function sends an echo back to the client
            conn.send(bytearray); % Echo
        end
        
        function onError(obj,conn,message)
            fprintf('%s\n',message)
        end
        
        function onClose(obj,conn,message)
            fprintf('%s\n',message)
        end
    end

end