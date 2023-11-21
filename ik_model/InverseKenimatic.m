joint1_damping = 0;
joint2_damping = 0;
damp_pince = 1000; % damping coefficient for joints of the pince
p_base = 'C:\study\projet_mt\digital_twin_robot\meshes\base_link.STL';

mdl = "robot_model";

load_system(mdl)

ik = simscape.multibody.KinematicsSolver(mdl);

base = "robot_model/World/W";
follower = "robot_model/gripper_base/F";
addFrameVariables(ik,"gripper_base","translation",base,follower);
addFrameVariables(ik,"gripper_base","rotation",base,follower);

targetIDs = ["gripper_base.Translation.x";"gripper_base.Translation.y";...
    "gripper_base.Translation.z"];
addTargetVariables(ik,targetIDs);
outputIDs =["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"];
addOutputVariables(ik,outputIDs);

guessesIDs = ["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"];
guesses = [0,0,0,0,0];
addInitialGuessVariables(ik,guessesIDs);


j1 = zeros(1000,1);
j2 = zeros(1000,1);
j3 = zeros(1000,1);
j4 = zeros(1000,1);
j5 = zeros(1000,1);
T = 10; % period
spline = zeros(1000,3);
% range for end ef
% fector space is aprroximately [-0.3,0.3] [-0.3,0.3] [0,0.3]
% here the trajectory we choose for the end effector is 
% x = 0.1*cos( t * 2*pi/T  ) y = 0.1*sin( t * 2*pi/T) z = 0.15 + 0.1 * t/10

for i = 1:1000
    targets = [0+0.1*cos(i/100*(2*pi/T)),0+0.1*sin(i/100*(2*pi/T)),0.15+0.1*(i/100/T)];
    spline(i,:) = targets;
    if i>1 
        guesses = [j1(i-1,1),j2(i-1,1),j3(i-1,1),j4(i-1,1),j5(i-1,1)];
    end
    [outputVec,statusFlag] = solve(ik,targets,guesses);
    j1(i,1) = outputVec(1);
    j2(i,1) = outputVec(2);
    j3(i,1) = outputVec(3);
    j4(i,1) = outputVec(4);
    j5(i,1) = outputVec(5);
    
end

joint1_ts = timeseries(j1/180*pi,0:0.01:9.99);
joint2_ts = timeseries(j2/180*pi,0:0.01:9.99);
joint3_ts = timeseries(j3/180*pi,0:0.01:9.99);
joint4_ts = timeseries(j4/180*pi,0:0.01:9.99);
joint5_ts = timeseries(j5/180*pi,0:0.01:9.99);

% pic_num = 1;
% for i = 1:100
%     subplot(5,1,1)
%     plot(1:1000,j1);
%     hold on
%     plot(10*i,j1(10*i),'o');
%     hold off
%     title('angle joint1')
%     subplot(5,1,2)
%     plot(1:1000,j2);
%     hold on
%     plot(10*i,j2(10*i),'o');
%     hold off
%     title('angle joint2')
%     subplot(5,1,3)
%     plot(1:1000,j3);
%     hold on
%     plot(10*i,j3(10*i),'o');
%     hold off
%     title('angle joint3')
%     subplot(5,1,4)
%     plot(1:1000,j4);
%     hold on
%     plot(10*i,j4(10*i),'o');
%     hold off
%     title('angle joint4')
%     subplot(5,1,5)
%     plot(1:1000,j5);
%     hold on
%     plot(10*i,j5(10*i),'o');
%     hold off
%     title('angle joint5')
%     F=getframe(gcf);
%     I=frame2im(F);
%     [I,map]=rgb2ind(I,256);
%     if pic_num == 1
%     imwrite(I,map,'test.gif','gif','Loopcount',inf,'DelayTime',0.1);
%     else
%     imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.1);
% 
%     end
% 
%     pic_num = pic_num + 1;
%     hold off
% end

