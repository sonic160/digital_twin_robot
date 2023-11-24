%Call the functions
%dataset=CreateDataset(shapes_dict,1000);
%small_circle_dataset = CreateCircleList(3, 3);
small_line_dataset =  Create_line_List(3,3);
%small_rectangle_dataset =  Create_rectangle_List(3,3,3);
%Display the result
%disp(small_circle_dataset)
disp(small_line_dataset)
%disp(small_rectangle_dataset)



%function1
function [R ]= calculateRotationMatrix(anglexy, anglexz, angleyz)
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


%Function2
function [newCell] = multiplyandsum(r,matrix1,Cell,matrix2)
newCell = cell(size(Cell));
   for i=1:numel(Cell)
       scaledfunction= @(x) r*(matrix1(i, 1) *Cell{1}(x) + matrix1(i, 2) * Cell{2}(x) + matrix1(i, 3) * Cell{3}(x))+matrix2(i);
       newCell{i}= scaledfunction;
   end
end



%function3
function [circlelist] = CreateCircleList(max_rayon,max_eloignement_centre)
        %il va falloir limitter les paramètres maximum de la génération de
        %form en fonction du mouvement permi par le bras
        circlelist=struct();
        x_prime_z_prime_y_prime_coords={@(t) cos(t);@(t) sin(t);@(t) 0};
        for r =0:max_rayon                                    %on itère sur les rayons possibles#changer incrémentation
            for e=0:max_eloignement_centre                    %on itère sur l'éloignement au centre possible #changer incrémentation?
                for anglexy=0:120:360                         %on explore les plans possibles en effectuant des rotations du plan xy autour de z
                    for anglexz=0:120:360                     %on explore les plans possibles en effectuant des rotations du plan xz autour de y
                        for angleyz=0:120:360                 %on explore les plans possibles en effectuant des rotations du plan yz autour de x          
                            for anglez=0:120:360              %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan z
                                for anglex=0:120:360          %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan x
                                    for angley=0:120:360      %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan y
                                        %coordonnées du centre du cercle
                                        %tracé
                                        ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
                                        Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
                                        circlecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);
                                        thiscircle=struct();
                                        circlename = sprintf('c_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r, e, anglexy, anglexz, angleyz, anglez, anglex, angley);
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



%Fonction 4
function [linelist] = Create_line_List(max_eloignement_centre,max_longueur)
        %il va falloir limitter les paramètres maximum de la génération de
        %form en fonction du mouvement permi par le bras
        linelist=struct();
        r=1;
        x_prime_z_prime_y_prime_coords={@(t) (t<=max_longueur)*t;@(t) 0;@(t) 0};
        for e=0:max_eloignement_centre                  %on itère sur l'éloignement au centre possible #changer incrémentation?
            for anglexy=0:120:360                       %on explore les plans possibles en effectuant des rotations du plan xy autour de z
                for anglexz=0:120:360                   %on explore les plans possibles en effectuant des rotations du plan xz autour de y
                    for angleyz=0:120:360%fait rien ici %on explore les plans possibles en effectuant des rotations du plan yz autour de x          
                        for anglez=0:120:360            %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan z
                            for anglex=0:120:360        %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan x
                                for angley=0:120:360    %on génère par incrément de 1 degré un "cercle" formé des centre des cercles que l'on va tracer à la distance voulue sur le plan y
                                    %coordonnées du centre du cercle
                                    %tracé
                                    ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
                                    Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
                                    linecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);
                                    thisline=struct();
                                    linename = sprintf('l_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r, e, anglexy, anglexz, angleyz, anglez, anglex, angley);
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


%Fonction 5
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