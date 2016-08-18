function model = read(path, fmt)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.2+
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %model = read(path, fmt)
    %
    %Read neural network model from given path. Supported are files written in either plain text or under mat compression.
    %
    %Parameters
    %----------
    %
    %path : str
    %the path to the file to read
    %
    %fmt : str
    %optional. explicitly state how to interpret the target file. if not given, format is inferred from path.
    %options are 'txt' for a plain text and 'mat' for producing a
    %mat-file
    %the plain text format is shared with the python implementation of the toolbox
    %
    %Returns
    %-------
    %model: modules.Sequential
    %the  neural network model, realized as a sequence of network modules.
    %
    %Notes
    %-----
    %the plain text file format is shared with the matlab implementation of the LRP Toolbox and describes
    %the model by listing its computational layers line by line as
    %
    %<Layername_i> [<input_size> <output_size>]
    %[<Layer_params_i>]
    %
    %since all implemented modules except for modules.Linear operate point-wise on the given data, the optional
    %information indicated by brackets [ ] is not used and only the name of the layer is witten, e.g.
    %
    %Rect
    %
    %Tanh
    %
    %SoftMax
    %
    %The exception formed by the linear layer implementation modules.Linear incorporates in raw text form as
    %
    %Linear m n
    %W(:)'
    %B(:)'
    %
    %with m and n being integer values describing the dimensions of the weight matrix W as [m x n] ,
    %W being the human readable ascii-representation of the flattened matrix (C-order) in m * n white space separated double values.
    %After the line describing W, the bias term B is written out as a single line of n white space separated double values.

    if ~exist(path,'file')
        throw(MException('MODEL_IO_READ:invalidPath',sprintf('model_io.read : No such file or directory: %s',path)))
    end

    if ~exist('fmt','var') %try to infer format
        [~,~,fmt] = fileparts(path);
        fmt(fmt == '.') = '';
        fmt=lower(fmt);
    end

    parsing_fxn = read_as(fmt);
    model = parsing_fxn(path);

end

function fxn = read_as(fmt)
    switch fmt
        case {'pickled', 'pickle','nn'}
            fxn = @read_pickled_unsupported;
        case {'npy, npz'}
            fxn = @read_np_unsupported;
        case 'txt'
            fxn = @read_txt;
        case 'mat'
            fxn = @read_mat;
        otherwise
            fxn = 'Unknown Format'
            fmt
    end
end

function read_pickled_unsupported(path)
    disp(['error parsing ' path])
    disp('reading pythons pickled file format not supported with matlab')
end

function read_np_unsupported(path)
    disp(['error parsing ' path])
    disp('reading numpy file format not supported with matlab')
end

function model = read_mat(path)
    disp(['loading mat-compressed model from ' path ])
    tmp = load(path);
    fn = fieldnames(tmp);
    model = eval(['tmp.' fn{1}]);
end

function model = read_txt(path)
    disp(['loading plain text model from ' path])

    function model = read_txt_helper(path)
        modools = {}; % avoid overloading the modules namespace
        fid = fopen(path);
        line = fgetl(fid);

        while ischar(line)
            if length(line) >= 6 && all(line(1:6) == 'Linear')
                lineparts = strsplit(line);
                m = str2double(lineparts{2});
                n = str2double(lineparts{3});

                layer = modules.Linear(m,n);
                %CAUTION HERE! matlab reshape order is different from numpy
                %reshape order!, thus the [n m] and transpose.
                layer.W = reshape(str2num(fgetl(fid)),[n m])';
                layer.B = str2num(fgetl(fid));
                modools{end+1} = layer;
            elseif length(line) == 4 && all(line(1:4) == 'Rect')
                modools{end+1} = modules.Rect();
            elseif length(line) == 4 && all(line(1:4) == 'Tanh')
                modools{end+1} = modules.Tanh();
            elseif length(line) == 7 && all(line(1:7) == 'SoftMax')
                modools{end+1} = modules.SoftMax();
            % TODO
            % elseif Convolution,Flatting,Pooling...
            else
                layername = strsplit(line);
                layername = layername{1};
                ERROR = MException('UnknownLayerType','Layer Type Identifyer %s not supported for reading from plain text file.',layername);
                throw(ERROR);
            end
            %read next line
            line = fgetl(fid);
        end %END WHILE

        model = modules.Sequential(modools);
    end % END read_txt_helper

    try
        model = read_txt_helper(path);
    catch ERROR
        % some error with parsing the text file has occurred at this point.
        % In this case: Try to fall back to the old plain text format.
        disp('probable reshaping / formatting error wile reading from plain text network file')
        disp(['Error Message: ' getReport(ERROR)])
        disp('Attempting fall-back to legacy plain text format interpretation...')
        model = read_txt_old(path);
        disp('fall-back successfull!')

    end
end


function model = read_txt_old(path)
    disp(['loading plain text model from ' path])

    modools = {}; % avoid overloading the modules namespace
    fid = fopen(path);
    line = fgetl(fid);

    while ischar(line)
        if length(line) >= 6 && all(line(1:6) == 'Linear')
            lineparts = strsplit(line);
            m = str2double(lineparts{2});
            n = str2double(lineparts{3});

            mod = modules.Linear(m,n);
            for i = 1:m
               mod.W(i,:) = str2num(fgetl(fid));
            end
            mod.B = str2num(fgetl(fid));
            modools{end+1} = mod;

        elseif length(line) == 4 && all(line(1:4) == 'Rect')
            modools{end+1} = modules.Rect();

        elseif length(line) == 4 && all(line(1:4) == 'Tanh')
            modools{end+1} = modules.Tanh();

        elseif length(line) == 7 && all(line(1:7) == 'SoftMax')
            modools{end+1} = modules.SoftMax();

        else
            layername = strsplit(line);
            layername = layername{1};
            ERROR = MException('UnknownLayerType','Layer Type Identifyer %s not supported for reading from plain text file.',layername);
            throw(ERROR);
        end

        line = fgetl(fid);
    end

    model = modules.Sequential(modools);

    fclose(fid);
end

