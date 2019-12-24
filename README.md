# 301 - Moved

MLIR is now part of LLVM, more information on https://mlir.llvm.org

The code from this repository can now be found at
https://github.com/llvm/llvm-project/tree/master/mlir/

# Migration

If you have a local fork of this repository or pull-requests that need to be
migrated to the LLVM monorepo, the following recipe may help you:

```
# From your local MLIR clone:
$ git clone git@github.com:newren/git-filter-repo.git /tmp/git-filter-repo
$ /tmp/git-filter-repo/git-filter-repo --path-rename :mlir/ --force  --message-callback 'return re.sub(b"(#[0-9]+)", b"tensorflow/mlir\\1", message)' --refs <branch name>
```

After this, all the commits from the previous upstream MLIR should match the
ones in the monorepo now. If you don't provide the `--refs` option, this
will rewrite *all the branches* in your repo.

From there you should be able to rebase any of your branch/commits on top of
the LLVM monorepo:

```
$ git remote set-url origin git@github.com:llvm/llvm-project.git
$ git fetch origin
$ git rebase origin/master -i
```

Cherry-picking commits should also work, if you checkout the master branch from
the monorepo you can `git cherry-pick <sha1>` from your (rewritten) branches.

You can also export patches with `git format-patch <range>` and re-apply it on
the monorepo using `git am <patch file>`.
