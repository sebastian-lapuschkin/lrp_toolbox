function data = read(path, fmt)
    % @author: Sebastian Lapuschkin
    % @maintainer: Sebastian Lapuschkin
    % @contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
    % @date: 14.08.2015
    % @version: 1.0
    % @copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
    % @license : BSD-2-Clause
    %
    %data = read(path, fmt)
    %
    %Read [N x D]-sized block-formatted data from a given path.
    %Supported data formats are
    %plain text (ascii-matrices)
    %matlab data files (mat-files)
    %
    %Parameters
    %----------
    %
    %path : str
    %the path to the file to read
    %
    %fmt : str
    %optional. if explicitly given, the file will be interpreted as mat, txt. elsewise, interpretation format will be inferred from the file name
    %
    %
    %Returns
    %-------
    %
    %data : [N x D]

    if ~exist(path,'file')
        throw(MException('DATA_IO_READ:invalidPath',sprintf('data_io.read : No such file or directory: %s',path)))
    end

    if ~exist('fmt','var') %try to infer format
        [~,~,fmt] = fileparts(path);
        fmt(fmt == '.') = '';
        fmt=lower(fmt);
    end

    parsing_fxn = read_as(fmt);
    data = parsing_fxn(path);

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

function data = read_mat(path)
    disp(['loading mat-compressed data from ' path ])
    tmp = load(path);
    fn = fieldnames(tmp);
    data = eval(['tmp.' fn{1}]);
end

function data = read_txt(path)
    disp(['loading plain text data from ' path])
    data = load(path, '-ascii');
end