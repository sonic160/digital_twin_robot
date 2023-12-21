

testlist=CreateRandomCircleList(0.28, 0.28,100);
disp(size(testlist));
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

function newCell = multiplyandsum(r,matrix1,Cell,matrix2)

newCell = cell(size(Cell));
   for i=1:numel(Cell)
      

       scaledfunction= @(x) r*(matrix1(i, 1) *Cell{1}(x) + matrix1(i, 2) * Cell{2}(x) + matrix1(i, 3) * Cell{3}(x))+matrix2(i);
       newCell{i}= scaledfunction;
   end
end

function randomIntegers = generateRandomNumbers(a, b, p)
    % Generate p random integers in the interval [a, b]
    randomIntegers = randi([round(a), round(b)], 1, p);
end

function [circlelist] = CreateRandomCircleList(max_rayon, max_eloignement_centre, num_trajectories)
    min_eloignement=0.02;
    max_eloignement=0.28;
    min_hauteur=0.1;
    circlelist=struct();
    x_prime_z_prime_y_prime_coords={@(t) cos(2*pi*t);@(t) sin(2*pi*t);@(t) 0};
    % Choose indices randomly
    e_values =generateRandomNumbers(min_hauteur*100, max_eloignement_centre*100, num_trajectories);
    r_values =generateRandomNumbers(1, max_rayon*100, num_trajectories);
    anglexy_values =generateRandomNumbers(1, 360, num_trajectories);
    anglexz_values =generateRandomNumbers(1, 360, num_trajectories);
    angleyz_values =generateRandomNumbers(1, 360, num_trajectories);
    anglez_values =generateRandomNumbers(1, 360, num_trajectories);
    anglex_values =generateRandomNumbers(1, 360, num_trajectories);
    angley_values =generateRandomNumbers(1, 360, num_trajectories);

    % Create trajectories
    for i = 1:num_trajectories
        %Choosing values
        e_h = e_values(i);
        r_h = r_values(i);
        e=e_h*0.01;
        r=r_h*0.01;
        %Filtering values that would cause clipping
        if (abs(e-r)<min_eloignement) || (e+r<min_eloignement) || (e+r>max_eloignement) || (abs(e-r)>max_eloignement)
                    continue
        end
        anglexy =anglexy_values(i);
        anglexz =anglexz_values(i);
        angleyz =angleyz_values(i);
        anglez =anglez_values(i);
        anglex =anglex_values(i);
        angley =angley_values(i);   
        %Writing the functions
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