import os, pkg_resources, shutil


def create_tutorials(destination, silent = False):
    """
    Copies mercury.monitoring tutorial notebooks to `destination`. A folder will be created inside
    destination, named 'monitoring_tutorials'. The folder `destination` must exist.

    Args:
        destination (str): The destination directory
        silent (bool): If True, suppresses output on success.

    Raises:
        ValueError: If `destination` is equal to source path.

    Examples:
        >>> # copy tutorials to /tmp/monitoring_tutorials
        >>> from mercury.monitoring import create_tutorials
        >>> create_tutorials('/tmp')

    """
    src = pkg_resources.resource_filename(__package__, 'tutorials')
    dst = os.path.abspath(destination)

    assert src != dst, 'Destination (%s) cannot be the same as source.' % src

    assert os.path.isdir(dst), 'Destination (%s) must be a directory.' % dst

    dst = os.path.join(dst, 'monitoring_tutorials')

    assert not os.path.exists(dst), 'Destination (%s) already exists' % dst

    shutil.copytree(src, dst)

    if not silent:
        print('Tutorials copied to: %s' % dst)
