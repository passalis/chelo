Development
===========

Contributing
------------

To contribute to the library:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes and add tests.
4. Submit a pull request with a detailed description of your changes.

Contribution Checklist
----------------------

Before submitting your pull request, ensure the following:

1. **Code Quality**:
   - Code follows the PEP 8 style guide.
   - All code is properly documented with clear comments.

2. **Testing**:
   - Tests cover new features or fixes.
   - All tests pass successfully using `pytest`.

3. **Documentation**:
   - Any new functionality is documented in the appropriate `.rst` files.
   - Update the `API Reference` section if new public classes or methods are added.

4. **Commit History**:
   - Each commit message is clear and descriptive.
   - Squash minor or fixup commits to keep history clean.

5. **File Formatting**:
   - Ensure no large binary files are included.
   - Check `.gitignore` for unwanted files.

6. **Compatibility**:
   - Ensure compatibility with Python 3.7 and above.
   - Verify that changes do not break existing functionality.

Testing
-------

To run tests:

.. code-block:: bash

   pytest tests/
