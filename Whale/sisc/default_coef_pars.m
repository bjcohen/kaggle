function coef_pars = default_coef_pars(coef_pars, varargin);

% DEFAULT_COEF_PARS Retrieves default parameter settings for computing activations.
%    Fields may be preset in argument PARS. These pre-settings may affect
%    the default values of other parameters. In addition, special parameters
%    may be passed as pairs. For example, to set pars.num_trials = 10000 and
%    set the special parameter 'verbosity' to 'silent', use
%
%        coef_pars.algorithm = 'lars';
%        coef_pars = default_pars(coef_pars,'verbosity','silent');
%
%
%   List of special parameters and their possible settings, default starred
%
%   exact                    Whether to compute exact solution for coefficients. (Default true)
%
%   verbosity
%     0                      Silent
%     1                      Print dot for every 10 iterations
%     2 (*)                  Print iteration number and objective function for each iteration
%     3-5                    Various levels of debugging information.
%
%   tile
%     true                   Divide X into overlapping tiles, and reconstruct each one.
%     false (*)
%
%
%  List of regular parameters
%
%   Coefficient search parameters
%    exact                   True if feature sign (C version) should find an exact solution.
%    num_coords              Number of coefficients to solve for in coordinate descent
%    coeff_iter              Maximum number of iterations when solving for one patch
%
%   Tiling parameters
%    tile                    See TILE above
%    tile_iters              Number of times to reconstruct all of the tiles
%
%   Display and save parameters
%    verbosity
%    dot_every               If verbosity == 1, display a dot after this many iterations.

if mod(nargin,2) == 0, error('Optional arguments must come in pairs.'); end;

special_pars = struct;

for curr=2:2:nargin-1,
  field = varargin{curr-1};
  val = varargin{curr};
  special_pars.(field) = val;
end

if ~isfield(special_pars, 'exact'), special_pars.exact = true; end;
if ~isfield(special_pars, 'verbosity'), special_pars.verbosity = 1; end;
if ~isfield(special_pars, 'tile'), special_pars.tile = false; end;

if ~isfield(coef_pars, 'dot_every'), coef_pars.dot_every = 10; end;
if ~isfield(coef_pars, 'verbosity'), coef_pars.verbosity = special_pars.verbosity; end;
if ~isfield(coef_pars, 'max_fs_tries'), coef_pars.max_fs_tries = 5; end;
if ~isfield(coef_pars, 'tile'), coef_pars.tile = special_pars.tile; end;
if coef_pars.tile,
  if ~isfield(coef_pars, 'tile_iters'), coef_pars.tile_iters = 2; end;
end

if special_pars.exact,
  if ~isfield(coef_pars, 'num_coords'), coef_pars.num_coords = 100; end;
  if ~isfield(coef_pars, 'coeff_iter'), coef_pars.coeff_iter = 10000; end;
  if ~isfield(coef_pars, 'exact'), coef_pars.exact = true; end;
else,  
  if ~isfield(coef_pars, 'num_coords'), coef_pars.num_coords = 300; end;
  if ~isfield(coef_pars, 'coeff_iter'), coef_pars.coeff_iter = 10; end;
  if ~isfield(coef_pars, 'exact'), coef_pars.exact = false; end;
end
