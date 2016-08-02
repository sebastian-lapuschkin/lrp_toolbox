function write(model, path, fmt)
	% @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
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

    if ~ isa(model, 'modules.Sequential') %TODO: get module string in a better way. matlab OOP sucks
        % TODO: ERROR HANDLING        
    end
    %TODO: increase precision for writing

    fid = fopen(path, 'wb');
    for i = 1:length(model.modules)
        mod = model.modules{i};
        if isa(mod, 'modules.Linear')
            fprintf(fid,'Linear %i %i\n',mod.m, mod.n);
            for ri = 1:size(mod.W,1)
                line = sprintf( '%e ',  mod.W(ri,:));
                line(end:end+1) = '\n';
                fprintf(fid, line);
            end
            line = sprintf( '%e ',  mod.B);
            line(end:end+1) = '\n';
            fprintf(fid, line);           

        else
            cname = class(mod);
            dots = find(cname == '.');
            fprintf(fid,[cname(dots+1:end) '\n']);
        end
    end
    fclose(fid);
end

