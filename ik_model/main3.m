
disp("It has begun")
%Creating shape set

 %adapted_circle_set=CreateCircleList(0.28, 0.28);
% adapted_line_set=CreateLineList(0.28, 0.28);
% adapted_shape_set=mergeStructures(adapted_circle_set,adapted_line_set);
% disp(adapted_shape_set)
% fieldnumbers=numel(fieldnames(adapted_shape_set));
% fprintf('The number of fields is:%d\n',fieldnumbers);

% reduced_adapted_circle_set= reduceStructureSize(adapted_circle_set, 500);
% reduced_adapted_line_set= reduceStructureSize(adapted_line_set, 500);
% reduced_adapted_shape_set= reduceStructureSize(adapted_shape_set, 500);
% disp(reduced_adapted_shape_set)
% fieldnumbers=numel(fieldnames(reduced_adapted_shape_set));
% fprintf('The number of fields is:%d\n',fieldnumbers);

%Clearing the trash
 clear adapted_circle_set;
% clear adapted_line_set;
% clear adapted_shape_set;

%Testing Shape_dict
% shapes_dict = struct(...
%     'xyCircle', struct('xequation', @(t) cos(t), 'yequation', @(t) sin(t), 'zequation', @(t) 0), ...
%     'xzCircle', struct('xequation', @(t) cos(t), 'yequation', @(t) 0, 'zequation', @(t) sin(t)), ...
%     'yzCircle', struct('xequation', @(t) 0, 'yequation', @(t) cos(t), 'zequation', @(t) sin(t)), ...
%     'xline', struct('xequation', @(t) t, 'yequation', @(t) 0, 'zequation', @(t) 0), ...
%     'yline', struct('xequation', @(t) 0, 'yequation', @(t) t, 'zequation', @(t) 0), ...
%     'zline', struct('xequation', @(t) 0, 'yequation', @(t) 0, 'zequation', @(t) t) ...
% );

%Loading Model

model_name = 'main3_armpi_fpv';
load_system(model_name);
joint1_damping = 0;
joint2_damping = 0;
damp_pince = 1000;
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
guesses = [3,3,3,3,3];
addInitialGuessVariables(ik,guessesIDs);


j1 = zeros(1000,1);
j2 = zeros(1000,1);
j3 = zeros(1000,1);
j4 = zeros(1000,1);
j5 = zeros(1000,1);
T = 10; % period
spline = zeros(1000,3);


%Creating dataset

%Selected shape_dict

shapes_dict=reduced_adapted_circle_set;
%shapes_dict=adapted_line_set;
%shapes_dict=reduced_adapted_shape_set;

len_time_series=1000;
m0=[zeros(1000, 1), zeros(1000, 1)];
m1=[transpose(1:1000), ones(1000, 1)];

shapelist=fieldnames(shapes_dict);
numberofshapes=numel(shapelist);
fprintf('The number of fields is:%d\n',numberofshapes);
dataset=[];
%for k = 1:numberofshapes
for k = 1:400
    sign=1;
    shape=shapelist{k};
    fprintf('Shape name:%s\n',shape);     
    %disp("Equations de la forme");
    %disp(shapes_dict.(shape));
    targets = zeros(len_time_series,3);

    for t = 1:len_time_series
        sign=1;
        t_echantillon=t/500;
        datapoint =[shapes_dict.(shape).xequation(t_echantillon), shapes_dict.(shape).yequation(t_echantillon), shapes_dict.(shape).zequation(t_echantillon)];
         %datapoint = [0+k*0.1*cos(t/100*(2*pi/T)),0+k*0.1*sin(t/100*(2*pi/T)),0.15+k*0.1*(t/100/T)];
            spline(t,:)  = datapoint;
            targets(t,:) = datapoint;

        
        
        if t>1 
            guesses = [j1(t-1,1),j2(t-1,1),j3(t-1,1),j4(t-1,1),j5(t-1,1)];
        end
    

        [outputVec,statusFlag] = solve(ik,datapoint,guesses);
        j1(t,1) = outputVec(1);
        j2(t,1) = outputVec(2);
        j3(t,1) = outputVec(3);
        j4(t,1) = outputVec(4);
        j5(t,1) = outputVec(5);
    end
        
    
    joint1_ts = timeseries(j1/180*pi,0:0.01:9.99);
    joint2_ts = timeseries(j2/180*pi,0:0.01:9.99);
    joint3_ts = timeseries(j3/180*pi,0:0.01:9.99);
    joint4_ts = timeseries(j4/180*pi,0:0.01:9.99);
    joint5_ts = timeseries(j5/180*pi,0:0.01:9.99);
    %dataset=[dataset,datapoint];
    for j=0:6   %on mettra 64 plus tard si on le veut
        fprintf('Motor off is:%d\n',j);
        error1=m1;
        error2=m1;
        error3=m1;
        error4=m1;
        error5=m1;
        error6=m1;
        switch j
            case 1
                error1=m0;
            case 2
                error2=m0;
            case 3
                error3=m0;
            case 4
                error4=m0;
            case 5
                error5=m0;
            case 6
                error6=m0;
        end
        dataset = [dataset, targets];
        %on ajoute déjà les trajectoires cibles
        disp("----------------")
        disp("----------------")
        simOut = sim(model_name);%je sais pas si error est vraiment pris en entrée ici ou non
        [x, y, z] = ForwardKinematic(j1, j2, j3, j4, j5);%inch ca dep vraiment de simOut
        jdatapoint = [x, y, z];%pour un j donné on met à la suite les len_time_series prédit  et les réels en prenant en compte le défault moteur, c'est ce qu'on donnera à manger à l'IA;
        dataset=[dataset,jdatapoint];
        disp(jdatapoint)
        %on ajoute jdatapoin au dataset, on a ainsi formé un bloc de six lignes associées à un point
    end
    %datapoint de labélisation j (pour une k ième forme donnée)
    %sera donnée par la mise bout a bout de 7*k i ème ligne de dataset
    %(cas pas d'erreure moteur)
    %et la 7*k+j ième ligne, chaque ligne étant x,y,z étudiés sur la
    %timeseries pour l'erreure moteur selecio
    %pour avoir une entrée de l'IA qui regarder l'effet  il faudra prendre
    %fprintf("The current size of the dataset is %s", mat2str(size(dataset)));
end
    
%fprintf("The final size of the dataset is %s", mat2str(size(dataset)));
disp("line 157 end has been passed");

%%experimental - mat2cell conversion %%%


% Specify the size of each submatrix (6x1000)

dataset = dataset';
clear size
sized = size(dataset);

rowDist = 6 * ones(1, sized(1)/6);
% Use mat2cell to convert the dataset into a cell array
cellArray = mat2cell(dataset, rowDist);
disp(size(cellArray))
% Now, cellArray is a cell array where each cell is a 6x1000 matrix



%%% end of experimental section %%%




function [x, y ,z] = ForwardKinematic(j1, j2, j3, j4, j5)
    joint1_damping = 0;
    joint2_damping = 0;
    damp_pince = 1000; % damping coefficient for joints of the pince
    
    mdl = "robot_model";
    
    load_system(mdl)
    
    ik = simscape.multibody.KinematicsSolver(mdl);
    
    base = "robot_model/World/W";
    follower = "robot_model/gripper_base/F";
    addFrameVariables(ik,"gripper_base","translation",base,follower);
    addFrameVariables(ik,"gripper_base","rotation",base,follower);
    
    targetIDs = ["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"] ;
    addTargetVariables(ik,targetIDs);
    outputIDs =["gripper_base.Translation.x";"gripper_base.Translation.y";...
        "gripper_base.Translation.z"];
    addOutputVariables(ik,outputIDs);
    
    x = zeros(1000,1);
    y = zeros(1000,1);
    z = zeros(1000,1);
    T = 10; % period
    spline = zeros(1000,5);
    
    
    for i = 1:1000
        targets = [j1(i),j2(i),j3(i),j4(i),j5(i)];
    
        spline(i,:) = targets;
    
        [outputVec,statusFlag] = solve(ik,targets);
        x(i,1) = outputVec(1);
        y(i,1) = outputVec(2);
        z(i,1) = outputVec(3);
    
        
    end
    



end

%function0
function reducedStruct = reduceStructureSize(inputStruct, n)
    % Get field names of the input structure
    fieldNames = fieldnames(inputStruct);
    
    % Check if n is greater than the number of fields
    if n > numel(fieldNames)
        error('n is greater than the number of fields in the input structure.');
    end
    
    % Randomly select n field names
    selectedFields = datasample(fieldNames, n, 'Replace', false);
    
    % Create the reduced structure
    reducedStruct = struct();
    for i = 1:numel(selectedFields)
        fieldName = selectedFields{i};
        reducedStruct.(fieldName) = inputStruct.(fieldName);
    end
end

%function1
function R = calculateRotationMatrix(anglexy, anglexz, angleyz)
    % Conversion des angles en radians
    anglexy = deg2rad(anglexy);
    anglexz = deg2rad(anglexz);
    angleyz = deg2rad(angleyz);

    % Matrices de rotation autour des axes
    Rz = [cos(anglexy) -sin(anglexy) 0; sin(anglexy) cos(anglexy) 0; 0 0 1];
    Rx = [1 0 0; 0 cos(anglexz) -sin(anglexz); 0 sin(anglexz) cos(anglexz)];
    Ry = [cos(angleyz) 0 sin(angleyz); 0 1 0; -sin(angleyz) 0 cos(angleyz)];

    % Calcul de la matrice totale de rotation
    R = Rz * Rx * Ry;
end


%function2
function newCell = multiplyandsum(r,matrix1,Cell,matrix2)

newCell = cell(size(Cell));
   for i=1:numel(Cell)
      

       scaledfunction= @(x) r*(matrix1(i, 1) *Cell{1}(x) + matrix1(i, 2) * Cell{2}(x) + matrix1(i, 3) * Cell{3}(x))+matrix2(i);
       newCell{i}= scaledfunction;
   end
end

%function3
function structure3 = mergeStructures(structure1, structure2)
    % Copy the contents of structure1 to structure3
    structure3 = structure1;
    
    % Get field names of structure2
    fields2 = fieldnames(structure2);

    % Iterate through fields of structure2 and add them to structure3
    for i = 1:length(fields2)
        field = fields2{i};
        structure3.(field) = structure2.(field);
    end
end



%function4
function [circlelist] = CreateCircleList(max_rayon,max_eloignement_centre)
        %il va falloir limitter les paramètres maximum de la génération de
        %form en fonction du mouvement permi par le bras
        %pas de 0.01 choisi dans premières boucles for experimentalement
        %valeures de min et max eloignement aussi
        %Valeures charac du robot
        min_eloignement=0.02;
        max_eloignement=0.28;
        circlelist=struct();
        x_prime_z_prime_y_prime_coords={@(t) cos(2*pi*t);@(t) sin(2*pi*t);@(t) 0};
        for e_h=0:1:max_eloignement_centre*100                      %on itère sur les rayons possibles#changer incrémentation
            e=e_h*0.01;
            for r_h =1:1:max_rayon*100                             %on itère sur l'éloignement au centre possible #changer incrémentation?
                r=r_h*0.01;
                %condition d'appliquabilité à la simulation,trop forte
                %vire des points qui marche mais peu de calculs
                if (abs(e-r)<min_eloignement) || (e+r<min_eloignement) || (e+r>max_eloignement) || (abs(e-r)>max_eloignement)
                    continue
                end
                for anglexy=0:180:360                         %on explore les plans possibles en effectuant des rotations du plan xy autour de z
                    for anglexz=0:180:360                     %on explore les plans possibles en effectuant des rotations du plan xz autour de y
                        for angleyz=0:180:360                 %on explore les plans possibles en effectuant des rotations du plan yz autour de x          
                            for anglez=0:180:360              %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan z
                                for anglex=0:180:360          %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan x
                                    for angley=0:180:360      %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan y
                                        %coordonnées du centre du cercle
                                        %tracé
                                        ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
                                        Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
                                        circlecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);
                                        thiscircle=struct();
                                        circlename = sprintf('c_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r_h, e_h, anglexy, anglexz, angleyz, anglez, anglex, angley);
                                        fieldName = sprintf('xequation');
                                        thiscircle.(fieldName)=circlecoords{1};
                                        fieldName = sprintf('yequation');
                                        thiscircle.(fieldName)=circlecoords{2};
                                        fieldName = sprintf('zequation');
                                        thiscircle.(fieldName)=circlecoords{3};
                                        circlelist.(circlename) = thiscircle;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
end



%function5
function [linelist] = CreateLineList(max_eloignement_centre,max_longueur)
        %il va falloir limitter les paramètres maximum de la génération de
        %form en fonction du mouvement permi par le bras
        linelist=struct();
        min_eloignement=0.02;
        max_eloignement=0.28;
        x_prime_z_prime_y_prime_coords={@(t) (t<=max_longueur)*t;@(t) 0;@(t) 0};
        for e_h=0:1:max_eloignement_centre*100         %on itère sur l'éloignement au centre possible #changer incrémentation?
            e=e_h*0.01;
            for r_h =1:1:max_longueur*100
                r=r_h*0.01;
                %condition d'appliquabilité à la simulation, trop forte
                %vire des points qui marche mais peu de calculs
                if (abs(e-r)<min_eloignement) || (e+r<min_eloignement) || (e+r>max_eloignement) || (abs(e-r)>max_eloignement)
                    continue
                end
                for anglexy=0:180:360                       %on explore les plans possibles en effectuant des rotations du plan xy autour de z
                    for anglexz=0:180:360                   %on explore les plans possibles en effectuant des rotations du plan xz autour de y
                        for angleyz=0:180:0%fait rien ici 0 %on explore les plans possibles en effectuant des rotations du plan yz autour de x          
                            for anglez=0:180:360            %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan z
                                for anglex=0:180:360        %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan x
                                    for angley=0:180:360    %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan y
                                        %coordonnées du centre du cercle
                                        %tracé
                                        ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
                                        Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
                                        linecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);
                                        thisline=struct();
                                        linename = sprintf('l_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r_h, e_h, anglexy, anglexz, angleyz, anglez, anglex, angley);
                                        fieldName = sprintf('xequation');
                                        thisline.(fieldName)=linecoords{1};
                                        fieldName = sprintf('yequation');
                                        thisline.(fieldName)=linecoords{2};
                                        fieldName = sprintf('zequation');
                                        thisline.(fieldName)=linecoords{3};
                                        linelist.(linename) = thisline;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
end


%function6
function [rectanglelist] = Create_rectangle_List(max_eloignement_centre,max_longueur,max_largeur)
        %il va falloir limitter les paramètres maximum de la génération de
        %form en fonction du mouvement permi par le bras
        rectanglelist=struct();
        r=1;
        x_prime_z_prime_y_prime_coords={@(t) (t<=max_longueur)*t-(max_longueur+max_largeur<t<=2*max_longueur+max_largeur)*(t-max_longueur+max_largeur);@(t) (max_longueur<t<=max_longueur+max_largeur)*(t-max_longueur) - (2*max_longueur+max_largeur<t<=2*max_longueur+2*max_largeur)*(t-2*max_longueur+max_largeur);@(t) 0};
        for e=0:max_eloignement_centre                  %on itère sur l'éloignement au centre possible #changer incrémentation?
            for anglexy=0:120:360                       %on explore les plans possibles en effectuant des rotations du plan xy autour de z
                for anglexz=0:120:360                   %on explore les plans possibles en effectuant des rotations du plan xz autour de y
                    for angleyz=0:120:360               %on explore les plans possibles en effectuant des rotations du plan yz autour de x          
                        for anglez=0:120:360            %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan z
                            for anglex=0:120:360        %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan x
                                for angley=0:120:360    %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan y
                                    %coordonnées du centre du cercle
                                    %tracé
                                    ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
                                    Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
                                    rectanglecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);
                                    thisrectangle=struct();
                                    rectanglename = sprintf('r_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r, e, anglexy, anglexz, angleyz, anglez, anglex, angley);
                                    fieldName = sprintf('xequation');
                                    thisrectangle.(fieldName)=rectanglecoords{1};
                                    fieldName = sprintf('yequation');
                                    thisrectangle.(fieldName)=rectanglecoords{2};
                                    fieldName = sprintf('zequation');
                                    thisrectangle.(fieldName)=rectanglecoords{3};
                                    rectanglelist.(rectanglename) = thisrectangle;
                                end
                            end
                        end
                    end
                end
            end
        end
end