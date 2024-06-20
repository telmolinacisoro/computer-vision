    function keypoints = find_extremas( dog, params )
    % FIND_EXTREMAS is a function that needs to take as input a computed difference of gaussian space and some parameters. 
    % It should return a a binary image where each found extrema has a value of 1 and the rest has value of 0.
    
    % define the search radius
    radius = 2;
    
    % intialize keypoints image to have the size of the original input image
    keypoints = zeros(size(dog{-params.omin+1}(:,:,1)));
    [mk, nk] = size (keypoints)
    for o = 1:params.O
        [M,N,S] = size(dog{o}) ;
        for s=2:S-2 
            for n=radius+1:N-radius
                for m=radius+1:M-radius
                    val = dog{o}(m,n,s);
                    neighbors = dog{o}(m-radius:m+radius, n-radius:n+radius, s-1:s+1);
                    neighbors = neighbors(neighbors ~= val);

    
                    if val > max(neighbors(:)) || val < min(neighbors(:))
                         % first transform the point to the coordinate space of the original image
                         % Compute D(x) = D + 0.5 * (gradient of D with respect to x) * x
                            dx = 0.5 * [dog{o}(m,n+1,s) - dog{o}(m,n-1,s); dog{o}(m+1,n,s) - dog{o}(m-1,n,s)]; % Gradient of D with respect to x
                            x = [n; m]; % Coordinates in the x direction
    
                            % Compute D(x^)
                            D_x_hat = val + dx.' * x;

                            if D_x_hat >= 0.03
                                ypt = round(m *2^(params.omin+o-1) ); 
                                xpt = round(n * 2^(params.omin+o-1) );
        
                                % take care of the boundaries
                                if(ypt < 1 ) 
                                    ypt = 1;
                                elseif (ypt > mk) 
                                        ypt = mk;
                                end
                                if (xpt <1 ) 
                                    xpt = 1;
                                elseif (xpt > nk) 
                                    xpt = nk;
                                end
        
			                     % set the values of keypoints to 1
                                keypoints(ypt, xpt) = 1;
                            end
		                 end
    
    
                end
            end
        end
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Helping Instructions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Once you find a minima or a maxima, you could use this set of intructions to set the value in the keypoints image to 1
    
    %                    if(is_minima || is_maxima)
    %                        % first transform the point to the coordinate space of the original image
    %                        ypt = round(y *2^(params.omin+o-1) ); 
    %                        xpt = round(x * 2^(params.omin+o-1) );
    %
    %                        % take care of the boundaries
    %                        if(ypt < 1 ) 
    %                            ypt = 1;
    %                        elseif (ypt > m) 
    %                                ypt = m;
    %                        end
    %                        if (xpt <1 ) 
    %                            xpt = 1;
    %                        elseif (xpt > n) 
    %                            xpt = n;
    %                        end
    %
    %			             % set the values of keypoints to 1
    %                        keypoints(ypt, xpt) = 1;
    %		             end
    
    
    end

