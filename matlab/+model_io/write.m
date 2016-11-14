function write(model, path, fmt)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.2+
    % @copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %write(model, path, fmt)
    %
    %neural a network model to a given path. Supported are either
    %plain text or mat files.
    %The model is cleaned of any temporary variables , e.g. hidden layer inputs or outputs, prior to writing
    %
    %Parameters
    %----------
    %
    %model : modules.Sequential
    %the object representing the model.
    %
    %path : str
    %the path to the file to read
    %
    %fmt : str
    %optional. explicitly state how to write the file. if not given, format is inferred from path.
    %options are 'txt' for a plain text and 'mat' to produce a mat
    %file
    %the plain text format is shared with the python implementation of the toolbox
    %
    %Notes
    %-----
    %see the Notes - Section in the function documentation of model_io.read for general info and a format
    %specification of the plain text representation of neural network models


    model.clean()
    if ~exist('fmt','var') %try to infer format
        [~,~,fmt] = fileparts(path);
        fmt(fmt == '.') = '';
        fmt=lower(fmt);
    end

    parsing_fxn = write_as(fmt);
    parsing_fxn(model, path);

end

function fxn = write_as(fmt)
    switch fmt
        case {'pickled', 'pickle','nn'}
            fxn = @write_pickled_unsupported;
        case {'npy, npz'}
            fxn = @write_np_unsupported;
        case 'txt'
            fxn = @write_txt;
        case 'mat'
            fxn = @write_mat;
        otherwise
            fxn = 'Unknown Format'
            fmt
    end
end

function write_pickled_unsupported(model,path)
    disp(['error parsing ' path])
    disp('writing pythons pickled file format not supported with matlab')
end

function write_np_unsupported(model,path)
    disp(['error parsing ' path])
    disp('writing numpy file format not supported with matlab')
end

function write_mat(model, path)
    disp(['writing model as mat-file to ' path])
    save(path,'model');
end

function write_txt(model,path)
    disp(['writing model as plain text to ' path])

    if ~ isa(model, 'modules.Sequential')
        throw(MException(['Argument "model" must be an instance of module.Sequential. wrapping a sequence of neural network computation layers, but is ' class(model)]));
    end
    %TODO: increase precision for writing ?

    fid = fopen(path, 'wb');
    for i = 1:length(model.modules)
        mod = model.modules{i};
        if isa(mod, 'modules.Linear')
            % Format of linear layer
            % Linear <rows_of_W> <columns_of_W>
            % <flattened (C-order) weight matrix W>
            % <flattened bias vector>
            fprintf(fid,'Linear %i %i\n',mod.m, mod.n);

            W = mod.W'; %transpose W to use C-order while flattening for compatibility to the python/numpy-implementation
            line = sprintf( '%e ',  W(:));  line(end:end+1) = '\n';
            fprintf(fid, line);

            line = sprintf( '%e ',  mod.B); line(end:end+1) = '\n';
            fprintf(fid, line);
        elseif isa(mod,'modules.Convolution')
            % Format of convolution layer
            % Convolution <rows_of_W> <columns_of_W> <depth_of_W> <number_of_filters_W> <stride_axis_1> <stride_axis_2>
            % <flattened (C-order) filter block W>
            % <flattened bias vector>
            hf = mod.filtersize(1);
            wf = mod.filtersize(2);
            df = mod.filtersize(3);
            nf = mod.filtersize(4);
            s1 = mod.stride(1);
            s2 = mod.stride(2);
            fprintf(fid,'Convolution %i %i %i %i %i %i\n',hf,wf,df,nf,s1,s2);
            
            W = permute(mod.W,[4 3 2 1]); %switch around axes to flatten out a C-order indexed version of mod.W
            line = sprintf( '%e ', W(:)); line(end:end+1) = '\n';
            fprintf(fid,line);
            
            line = sprintf( '%e ',  mod.B(:)); line(end:end+1) = '\n';
            fprintf(fid, line);
        elseif isa(mod,'modules.SumPool')
            % Format of sum pooling layer
            % SumPool <mask_heigth> <mask_width> <stride_axis_1> <stride_axis_2>
            hpool = mod.pool(1);
            wpool = mod.pool(2);
            s1 = mod.stride(1);
            s2 = mod.stride(2);
            fprintf(fid,'SumPool %i %i %i %i\n',hpool,wpool,s1,s2);
        elseif isa(mod, 'modules.MaxPool')
            % Format of max pooling layer
            % MaxPool <mask_heigth> <mask_width> <stride_axis_1> <stride_axis_2>
            hpool = mod.pool(1);
            wpool = mod.pool(2);
            s1 = mod.stride(1);
            s2 = mod.stride(2);
            fprintf(fid,'MaxPool %i %i %i %i\n',hpool,wpool,s1,s2);
        else
            % all other layers are free from parameters. Format is thus:
            % <Layername>

            cname = class(mod);
            dots = find(cname == '.');
            fprintf(fid,[cname(dots+1:end) '\n']);
        end
    end
    fclose(fid);
end

