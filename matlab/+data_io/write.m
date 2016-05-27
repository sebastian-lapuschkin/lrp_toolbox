function write(data, path, fmt)
    % @author: Sebastian Bach
    % @maintainer: Sebastian Bach
    % @contact: sebastian.bach@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
    % @license : BSD-2-Clause
    %
    %write(data, path, fmt)
    %
    %Write [N x D]-sized block-formatted data to a given path.
    %Supported data formats are
    %plain text (ascii-matrices)
    %matlab data files (mat-files)
    %
    %Parameters
    %----------
    %
    %data : [N x D]
    %matrix of data
    %
    %path : str
    %the path to write the data to
    %
    %fmt : str
    %optional. if explicitly given, the file will be written as mat or txt. elsewise, interpretation format will be inferred from the file name


    if ~exist('fmt','var') %try to infer format
        [~,~,fmt] = fileparts(path);
        fmt(fmt == '.') = '';
        fmt=lower(fmt);
    end

    parsing_fxn = write_as(fmt);
    parsing_fxn(data,path);

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


function write_pickled_unsupported(data,path)
    disp(['error parsing ' path])
    disp('writing pythons pickled file format not supported with matlab')
end

function write_np_unsupported(data,path)
    disp(['error parsing ' path])
    disp('writing numpy file format not supported with matlab')
end

function write_mat(data,path)
    disp(['writing mat-compressed data to ' path ])
    save(path, 'data');
end

function  write_txt(data,path)
    disp(['writing plain text data to ' path])
    save(path, 'data', '-ascii');
end