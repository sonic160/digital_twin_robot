

testlist=CreateRandomLineList(0.28, 0.28,100);
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

function [linelist] = CreateRandomLineList(max_rayon, max_eloignement_centre, num_trajectories)
    min_eloignement = 0.02;
    max_eloignement = 0.28;
    min_hauteur = 0.1;
    linelist=struct();
    x_prime_z_prime_y_prime_coords={@(t) (t<=max_longueur)*t;@(t) 0;@(t) 0};
    % Initialize counter
    generated_trajectories = 0;

    % Keep generating trajectories until the desired number is reached
    while generated_trajectories < num_trajectories
        % Choose indices randomly
        e_values = generateRandomNumbers(min_hauteur*100, max_eloignement_centre*100, 1);
        r_values = generateRandomNumbers(1, max_rayon*100, 1);
        anglexy_values = generateRandomNumbers(1, 360, 1);
        anglexz_values = generateRandomNumbers(1, 360, 1);
        angleyz_values = generateRandomNumbers(1, 360, 1);
        anglez_values = generateRandomNumbers(1, 360, 1);
        anglex_values = generateRandomNumbers(1, 360, 1);
        angley_values = generateRandomNumbers(1, 360, 1);

        % Create a single trajectory
        for i = 1:1
            % Choosing values
            e_h = e_values(i);
            r_h = r_values(i);
            e = e_h * 0.01;
            r = r_h * 0.01;

            % Filtering values that would cause clipping
            if (abs(e-r) < min_eloignement) || (e+r < min_eloignement) || (e+r > max_eloignement) || (abs(e-r) > max_eloignement)
                continue
            end

            anglexy = anglexy_values(i);
            anglexz = anglexz_values(i);
            angleyz = angleyz_values(i);
            anglez = anglez_values(i);
            anglex = anglex_values(i);
            angley = angley_values(i);

            % Writing the functions
            ecoord=e*calculateRotationMatrix(anglez, angley, anglex)*[1;1;1];
            Rotation=calculateRotationMatrix(anglexy, anglexz, angleyz);
            linecoords=multiplyandsum(r,Rotation,x_prime_z_prime_y_prime_coords,ecoord);

            % Create a structure for this trajectory
            thisline=struct();
            linename = sprintf('l_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r_h, e_h, anglexy, anglexz, angleyz, anglez, anglex, angley);
            fieldName = sprintf('xequation');
            thisline.(fieldName)=linecoords{1};
            fieldName = sprintf('yequation');
            thisline.(fieldName)=linecoords{2};
            fieldName = sprintf('zequation');
            thisline.(fieldName)=linecoords{3};

            % Add this trajectory to the structure
            linelist.(linename) = thisline;

            % Increment the counter
            generated_trajectories = generated_trajectories + 1;
        end
    end
end

function [circlelist] = CreateRandomCircleList(max_rayon, max_eloignement_centre, num_trajectories)
    min_eloignement = 0.02;
    max_eloignement = 0.28;
    min_hauteur = 0.1;
    circlelist = struct();
    x_prime_z_prime_y_prime_coords = {@(t) cos(2*pi*t); @(t) sin(2*pi*t); @(t) 0};

    % Initialize counter
    generated_trajectories = 0;

    % Keep generating trajectories until the desired number is reached
    while generated_trajectories < num_trajectories
        % Choose indices randomly
        e_values = generateRandomNumbers(min_hauteur*100, max_eloignement_centre*100, 1);
        r_values = generateRandomNumbers(1, max_rayon*100, 1);
        anglexy_values = generateRandomNumbers(1, 360, 1);
        anglexz_values = generateRandomNumbers(1, 360, 1);
        angleyz_values = generateRandomNumbers(1, 360, 1);
        anglez_values = generateRandomNumbers(1, 360, 1);
        anglex_values = generateRandomNumbers(1, 360, 1);
        angley_values = generateRandomNumbers(1, 360, 1);

        % Create a single trajectory
        for i = 1:1
            % Choosing values
            e_h = e_values(i);
            r_h = r_values(i);
            e = e_h * 0.01;
            r = r_h * 0.01;

            % Filtering values that would cause clipping
            if (abs(e-r) < min_eloignement) || (e+r < min_eloignement) || (e+r > max_eloignement) || (abs(e-r) > max_eloignement)
                continue
            end

            anglexy = anglexy_values(i);
            anglexz = anglexz_values(i);
            angleyz = angleyz_values(i);
            anglez = anglez_values(i);
            anglex = anglex_values(i);
            angley = angley_values(i);

            % Writing the functions
            ecoord = e * calculateRotationMatrix(anglez, angley, anglex) * [1; 1; 1];
            Rotation = calculateRotationMatrix(anglexy, anglexz, angleyz);
            circlecoords = multiplyandsum(r, Rotation, x_prime_z_prime_y_prime_coords, ecoord);

            % Create a structure for this trajectory
            thiscircle = struct();
            circlename = sprintf('c_r%d_e%d_xy%d_xz%d_yz%d_z%d_x%d_y%d', r_h, e_h, anglexy, anglexz, angleyz, anglez, anglex, angley);
            fieldName = sprintf('xequation');
            thiscircle.(fieldName) = circlecoords{1};
            fieldName = sprintf('yequation');
            thiscircle.(fieldName) = circlecoords{2};
            fieldName = sprintf('zequation');
            thiscircle.(fieldName) = circlecoords{3};

            % Add this trajectory to the structure
            circlelist.(circlename) = thiscircle;

            % Increment the counter
            generated_trajectories = generated_trajectories + 1;
        end
    end
end