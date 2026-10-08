$ENV{'TEXINPUTS'} = './styles//:./figures//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = './styles//:' . ($ENV{'BSTINPUTS'} // '');
$ENV{'BIBINPUTS'} = './/:' . ($ENV{'BIBINPUTS'} // '');
$out_dir = 'build';
