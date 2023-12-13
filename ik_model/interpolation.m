rayon = 0.28;
height = 0.28;
num_point = 4; % number of point for interpolation
flag = 1;

while flag ~= 0
    flag = 0;
    m = rand(3,num_point); %each point represent by a column 
    m(1,:) = m(1,:)*rayon*2-rayon;%x
    m(2,:) = m(2,:)*rayon*2-rayon;%y
    m(3,:) = m(3,:)*height;%z
    for i = 1:num_point
        if m(:,i)'*m(:,i)>= rayon^2
            flag = 1;
        end
    end
end
X = m(1,:);Y = m(2,:);Z = m(3,:);
values = spcrv([X(1) X X(end);Y(1) Y Y(end);Z(1) Z Z(end)],4);
plot3(X,Y,Z)
hold on
plot3(values(1,:),values(2,:),values(3,:))

ts_x = timeseries(values(1,:),linspace(0,10,size(values,2)));
ts_y = timeseries(values(2,:),linspace(0,10,size(values,2)));
ts_z = timeseries(values(3,:),linspace(0,10,size(values,2)));

ts_x = resample(ts_x,0.01:0.01:10);

ts_y = resample(ts_y,0.01:0.01:10);
ts_z = resample(ts_z,0.01:0.01:10);

%plot3(reshape(ts_x.data,[1 1000]),reshape(ts_y.data,[1 1000]),reshape(ts_z.data,[1 1000]))
