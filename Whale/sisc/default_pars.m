function pars = default_pars(pars,varargin)

% DEFAULT_PARS Retrieves default parameter settings for running the code on images.
%    Fields may be preset in argument PARS. These pre-settings may affect
%    the default values of other parameters. In addition, special parameters
%    may be passed as pairs. For example, to set pars.num_trials = 10000 and
%    set the special parameter 'mode' to 'learn', use
%
%        pars.num_trials = 10000;
%        pars = default_pars(pars, 'save', false);
%
%  List of special parameters and their settings
%
%   save
%     true
%     false
%
%
%  List of regular parameters
%
%   Image parameters
%    num_trials              Number of batches to solve for
%    batch_size              Number of patches in a batch
%    test_fobj               If true, compute objective function on test set for checking convergence
%    patch_M                 Height of patch (default = patch_dim)
%    patch_N                 Width of patch (default = patch_dim)
%    crop_amt                Crop this much off the edge of images
%    overlap_M               Amount by which patches are allowed to overlap vertically
%    overlap_N               Amount by which patches are allowed to overlap horizontally
%
%   Objective function parameters
%     beta                   Sparsity penalty
%
%   Basis parameters
%    num_bases               
%    basis_M                 Height of bases (default = basis_dim)
%    basis_N                 Width of bases (default = basis_dim)
%
%   Display & save parameters
%    display_bases_every     Display bases after this many batches
%    save_bases_every        Save bases, stats, and possibly all of the coefficients
%                              after doing this many batches
%    basedir
%    verbosity
%    savename

if mod(nargin,2) == 0, error('Optional arguments must come in pairs.'); end;

special_pars = struct;

for curr=2:nargin-1,
  field = varargin{curr-1};
  val = varargin{curr};
  special_pars.(field) = val;
end

if ~isfield(special_pars, 'save'), special_pars.save = false; end;

if ~isfield(pars, 'verbosity'), pars.verbosity = 2; end;
if ~isfield(pars, 'save_stats'), pars.save_stats = true; end;
if ~isfield(pars, 'display_bases_every'), pars.display_bases_every = 1; end;

if ~isfield(pars, 'test_fobj'), pars.test_fobj = true; end;
if ~isfield(pars, 'num_test'), pars.num_test = 100; end;

if ~isfield(pars, 'num_trials'), pars.num_trials = 400; end;
if ~isfield(pars, 'batch_size'), pars.batch_size = 200; end;
if ~isfield(pars, 'patch_M'), pars.patch_M = 80; end;
if ~isfield(pars, 'patch_N'), pars.patch_N = 80; end;
if ~isfield(pars, 'basis_M'), pars.basis_M = 24; end;
if ~isfield(pars, 'basis_N'), pars.basis_N = 24; end;
if ~isfield(pars, 'beta'), pars.beta = 200; end;
if special_pars.save,
  if ~isfield(pars, 'basedir') || ~isfield(pars, 'savename'), error; end;
  if ~isfield(pars, 'save_bases_every'), pars.save_bases_every = 1; end;
else
  if ~isfield(pars, 'save_bases_every'), pars.save_bases_every = 0; end;
end

if ~isfield(pars, 'overlap_M'), pars.overlap_M = pars.basis_M; end;
if ~isfield(pars, 'overlap_N'), pars.overlap_N = pars.basis_N; end;

if ~isfield(pars, 'num_bases'), pars.num_bases = 16; end;
if ~isfield(pars, 'dot_every'), pars.dot_every = 10; end;

if ~isfield(pars, 'crop_amt'), pars.crop_amt = 4; end;